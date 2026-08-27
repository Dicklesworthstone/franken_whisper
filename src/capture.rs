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
const PIPE_CHILD_EXIT_WAIT: Duration = Duration::from_secs(1);
/// Hard memory envelope for any capture ring. This still admits five minutes
/// of 48 kHz stereo f32 audio (28.8M samples) with headroom, while preventing
/// caller- or device-supplied rate/channel values from becoming an unbounded
/// allocation request.
const MAX_CAPTURE_RING_SAMPLES: usize = 32 * 1024 * 1024;
/// Multichannel downmix is useful for real audio layouts, but a raw u16 channel
/// count is not a safe allocation/CPU contract for a streaming resampler.
const MAX_RESAMPLER_CHANNELS: u16 = 64;
/// Bound per-output CPU independently of total filter storage. Common rates
/// through 384 kHz remain below this ceiling; pathological decimation ratios
/// cannot turn each 16 kHz output sample into millions of MACs.
const MAX_RESAMPLER_TAPS_PER_PHASE: usize = 4 * 1024;
/// The polyphase filter is retained for the life of the stream. Sixteen MiB of
/// f32 coefficients is already far beyond common 44.1/48/96/192 kHz layouts;
/// larger rational ratios are rejected before allocation.
const MAX_RESAMPLER_FILTER_TAPS: usize = 4 * 1024 * 1024;

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
    /// audio then reports `ended: true`. Cleanup failures are returned so a
    /// live session cannot report success while an owned worker survives.
    fn stop(&mut self) -> FwResult<()>;
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

/// Resolve a bounded ring size before allocating or starting a producer.
/// Shared by hardware and pipe sources so every public capture constructor
/// enforces the same memory envelope.
pub(crate) fn capture_ring_capacity_samples(
    sample_rate: u32,
    channels: u16,
    ring_capacity_sec: f64,
) -> FwResult<usize> {
    if sample_rate == 0 || channels == 0 {
        return Err(FwError::InvalidRequest(
            "capture ring requires nonzero sample rate and channel count".to_owned(),
        ));
    }
    if !ring_capacity_sec.is_finite() || ring_capacity_sec <= 0.0 {
        return Err(FwError::InvalidRequest(
            "capture ring capacity must be finite and greater than zero seconds".to_owned(),
        ));
    }

    let channel_count = usize::from(channels);
    let minimum = channel_count.checked_mul(1024).ok_or_else(|| {
        FwError::InvalidRequest("capture ring minimum size overflows usize".to_owned())
    })?;
    let requested = f64::from(sample_rate) * ring_capacity_sec * f64::from(channels);
    if !requested.is_finite()
        || requested > MAX_CAPTURE_RING_SAMPLES as f64
        || minimum > MAX_CAPTURE_RING_SAMPLES
    {
        return Err(FwError::InvalidRequest(format!(
            "capture ring requests {requested:.0} samples for {sample_rate} Hz/{channels}ch/{ring_capacity_sec}s; maximum is {MAX_CAPTURE_RING_SAMPLES}"
        )));
    }
    Ok((requested.ceil() as usize).max(minimum))
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
                    // Actionable taxonomy (bd-rt-device-probe-wh02): name the
                    // request AND what exists.
                    let _ = setup_tx.send(CpalSetup::Failed(
                        device_not_found_error(requested).to_string(),
                    ));
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

                let capacity_samples = match capture_ring_capacity_samples(
                    sample_rate,
                    channels,
                    ring_capacity_sec,
                ) {
                    Ok(capacity) => capacity,
                    Err(error) => {
                        let _ = setup_tx.send(CpalSetup::Failed(error.to_string()));
                        return;
                    }
                };
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

    fn stop(&mut self) -> FwResult<()> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        let mut join_error = None;
        if let Some(join) = self.join.take() {
            if join.join().is_err() {
                join_error = Some(FwError::ContractViolation(
                    "cpal capture worker panicked during shutdown".to_owned(),
                ));
            }
        }
        self.shared.ended.store(true, Ordering::Release);
        match join_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for CpalCaptureSource {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            tracing::warn!(%error, "cpal capture cleanup failed during drop");
        }
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

    fn stop(&mut self) -> FwResult<()> {
        self.stopped = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// bd-rt-device-probe-wh02: input-device discovery + failure taxonomy
//
// Agents diagnosing "why is the mic silent" need structured answers, not
// stderr archaeology. Three pieces live here: enumeration (metadata only —
// NEVER opens a stream, because opening triggers the macOS TCC microphone
// prompt as a side effect, which is unacceptable for a health probe),
// the not-found error that lists what IS available, and a silent-input
// watchdog the driver arms at session start.
// ---------------------------------------------------------------------------

/// Metadata for one input device (enumeration only; no stream is opened).
#[derive(Debug, Clone, PartialEq)]
pub struct InputDeviceInfo {
    pub name: String,
    pub is_default: bool,
    /// Default-config sample rate (Hz) when the device reports one.
    pub default_sample_rate_hz: Option<u32>,
    /// Default-config channel count when the device reports one.
    pub default_channels: Option<u16>,
}

/// Enumerate input devices, deterministically ordered (default device
/// first, then by name). Metadata queries only — no TCC prompt.
pub fn enumerate_input_devices() -> FwResult<Vec<InputDeviceInfo>> {
    use cpal::traits::{DeviceTrait as _, HostTrait as _};
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let mut devices: Vec<InputDeviceInfo> = Vec::new();
    let iter = host.input_devices().map_err(|error| {
        FwError::BackendUnavailable(format!("cannot enumerate audio input devices: {error}"))
    })?;
    for device in iter {
        let Ok(name) = device.name() else { continue };
        let default_config = device.default_input_config().ok();
        devices.push(InputDeviceInfo {
            is_default: default_name.as_deref() == Some(name.as_str()),
            default_sample_rate_hz: default_config.as_ref().map(|c| c.sample_rate().0),
            default_channels: default_config
                .as_ref()
                .map(cpal::SupportedStreamConfig::channels),
            name,
        });
    }
    devices.sort_by(|a, b| {
        b.is_default
            .cmp(&a.is_default)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(devices)
}

/// macOS TCC remediation text (surfaced whenever capture opens but delivers
/// nothing, or the platform denies the stream).
pub const MIC_PERMISSION_REMEDIATION: &str = "if no audio arrives: System Settings > Privacy & \
     Security > Microphone (macOS; terminal apps prompt on first use — over SSH there is no \
     prompt, pre-authorize or use --source stdin-pcm)";

/// The device-not-found error, naming the requested device AND what exists
/// (the actionable form the bead's taxonomy requires).
pub fn device_not_found_error(requested: &str) -> FwError {
    let known = enumerate_input_devices()
        .map(|devices| {
            devices
                .iter()
                .map(|d| d.name.as_str().to_owned())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|_| "<enumeration unavailable>".to_owned());
    FwError::InvalidRequest(format!(
        "input device not found: `{requested}`; available input devices: [{known}]"
    ))
}

/// Silent-input watchdog (driver arms it at session start): triggers ONCE
/// when the first `threshold_sec` seconds of a session are PURE digital
/// zeros — the signature of an unauthorized mic on macOS (TCC denial
/// delivers zeros, not errors). Any nonzero sample disarms it permanently;
/// genuine low-level room noise (even -90 dBFS dither) never triggers it.
/// The driver surfaces the trigger as `listen.warning {reason:
/// "silent_input"}` and keeps the session alive (some devices legitimately
/// start silent).
#[derive(Debug)]
pub struct SilentInputWatchdog {
    threshold_samples: u64,
    zero_samples: u64,
    disarmed: bool,
    fired: bool,
}

impl SilentInputWatchdog {
    #[must_use]
    pub fn new(threshold_sec: f64, sample_rate: u32) -> Self {
        Self {
            threshold_samples: (threshold_sec * f64::from(sample_rate)).round() as u64,
            zero_samples: 0,
            disarmed: false,
            fired: false,
        }
    }

    /// Feed session audio; returns `true` exactly once, at the moment the
    /// leading-zeros threshold is crossed.
    pub fn observe(&mut self, samples: &[f32]) -> bool {
        if self.disarmed || self.fired {
            return false;
        }
        for &sample in samples {
            if sample != 0.0 {
                self.disarmed = true;
                return false;
            }
            self.zero_samples += 1;
            if self.zero_samples >= self.threshold_samples {
                self.fired = true;
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// bd-rt-ffmpeg-pipe-7dbu: pipe-fed capture sources
//
// Two producers share one generic core: `StdinPcmSource` (raw PCM piped from
// ANY upstream — `ffmpeg -f s16le -`, sox, a remote shell — the composable
// path and the latency harness's instrument) and `FfmpegCaptureSource` (the
// device-capture fallback when cpal cannot serve). Both push into the same
// SPSC ring/`SharedState` machinery as the cpal source, so the driver's
// pull-side contract is identical across all sources.
// ---------------------------------------------------------------------------

/// Raw-PCM wire formats accepted on stdin (`--stdin-format`). Anything else
/// is FW-INVALID-REQUEST naming these two — fixed-size frames keep the
/// chunk-reassembly logic trivial and the contract auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    /// Little-endian signed 16-bit (ffmpeg `-f s16le`). The default.
    S16le,
    /// Little-endian IEEE f32 (`-f f32le`) — what most Rust/Python audio
    /// tools emit natively.
    F32le,
}

impl PcmFormat {
    #[must_use]
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16le => 2,
            Self::F32le => 4,
        }
    }

    /// Parse a `--stdin-format` value.
    pub fn parse(value: &str) -> FwResult<Self> {
        match value {
            "s16le" => Ok(Self::S16le),
            "f32le" => Ok(Self::F32le),
            other => Err(FwError::InvalidRequest(format!(
                "unsupported stdin PCM format `{other}`; supported: s16le, f32le"
            ))),
        }
    }
}

/// Generic reader-thread capture source over any byte stream delivering raw
/// interleaved PCM. Sample values straddling chunk boundaries are carried
/// (byte-level reassembly), conversion is exact (`i16 / 32768`, f32 verbatim).
impl std::fmt::Debug for PipePcmSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipePcmSource")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

pub struct PipePcmSource {
    consumer: ringbuf::HeapCons<f32>,
    shared: Arc<SharedState>,
    join: Option<std::thread::JoinHandle<()>>,
    /// Child owner when this pipe is a subprocess's stdout (kill-on-stop).
    child: Option<crate::process::StreamingChild>,
    /// Only an owned child gives us authority to close the reader's pipe.
    /// Externally owned readers such as stdin must be detached on stop rather
    /// than joined while another process can keep them blocked indefinitely.
    join_reader_on_stop: bool,
    sample_rate: u32,
    channels: u16,
}

impl PipePcmSource {
    /// Build over an arbitrary reader (tests use in-memory pipes; stdin and
    /// ffmpeg-stdout are the production callers).
    pub fn from_reader<R>(
        reader: R,
        format: PcmFormat,
        sample_rate: u32,
        channels: u16,
        ring_capacity_sec: f64,
    ) -> FwResult<Self>
    where
        R: std::io::Read + Send + 'static,
    {
        let capacity_samples =
            capture_ring_capacity_samples(sample_rate, channels, ring_capacity_sec)?;
        let ring = HeapRb::<f32>::new(capacity_samples);
        let (mut producer, consumer) = ring.split();
        let shared = SharedState::new();
        let shared_reader = Arc::clone(&shared);
        let ch = usize::from(channels);
        let join = std::thread::Builder::new()
            .name("fw-capture-pipe".into())
            .spawn(move || {
                let mut reader = reader;
                let bps = format.bytes_per_sample();
                let mut raw = vec![0u8; 32 * 1024];
                let mut carry: Vec<u8> = Vec::with_capacity(bps);
                // Decoded samples not yet pushed (persists across reads so a
                // frame split by a chunk boundary is completed, never lost).
                let mut pending: Vec<f32> = Vec::with_capacity(raw.len() / bps + 8);
                loop {
                    if shared_reader.ended.load(Ordering::Acquire) {
                        break; // stop() requested
                    }
                    match std::io::Read::read(&mut reader, &mut raw) {
                        Ok(0) => {
                            let trailing_bytes = carry.len() + pending.len() * bps;
                            if trailing_bytes > 0 {
                                shared_reader.record_error(format!(
                                    "pipe capture ended with incomplete PCM frame: {trailing_bytes} trailing byte(s); expected a multiple of {} bytes for {channels} channel(s) of {format:?}",
                                    bps * ch
                                ));
                            }
                            break;
                        }
                        Err(error) => {
                            if error.kind() != std::io::ErrorKind::Interrupted {
                                shared_reader
                                    .record_error(format!("pipe capture read failed: {error}"));
                                break;
                            }
                        }
                        Ok(n) => {
                            let mut bytes: &[u8] = &raw[..n];
                            // Complete a carried partial sample first.
                            if !carry.is_empty() {
                                let need = bps - carry.len();
                                let take = need.min(bytes.len());
                                carry.extend_from_slice(&bytes[..take]);
                                bytes = &bytes[take..];
                                if carry.len() == bps {
                                    pending.push(decode_sample(format, &carry));
                                    carry.clear();
                                }
                            }
                            let whole = bytes.len() - bytes.len() % bps;
                            for sample in bytes[..whole].chunks_exact(bps) {
                                pending.push(decode_sample(format, sample));
                            }
                            carry.extend_from_slice(&bytes[whole..]);
                            // Push whole frames only; samples of a frame split
                            // by the chunk boundary stay pending for the next
                            // read (never lost, never misaligned).
                            let aligned = pending.len() - pending.len() % ch;
                            let dropped = feed_ring(&mut producer, &pending[..aligned], ch);
                            if dropped > 0 {
                                shared_reader
                                    .overrun_frames
                                    .fetch_add(dropped, Ordering::Relaxed);
                            }
                            pending.drain(..aligned);
                        }
                    }
                }
                shared_reader.ended.store(true, Ordering::Release);
                tracing::debug!("pipe capture reader thread exited");
            })
            .map_err(FwError::Io)?;
        Ok(Self {
            consumer,
            shared,
            join: Some(join),
            child: None,
            join_reader_on_stop: false,
            sample_rate,
            channels,
        })
    }

    /// Attach the owning child process (killed on stop/drop).
    fn with_child(mut self, child: crate::process::StreamingChild) -> Self {
        self.child = Some(child);
        self.join_reader_on_stop = true;
        self
    }
}

impl CaptureSource for PipePcmSource {
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
        let usable = out.len() - out.len() % ch;
        let deadline = Instant::now() + max_wait;
        let mut written = 0usize;
        loop {
            written += self.consumer.pop_slice(&mut out[written..usable]);
            let ended_flag = self.shared.ended.load(Ordering::Acquire);
            if written > 0 || Instant::now() >= deadline || ended_flag {
                if let Some(message) = self.shared.take_error() {
                    // Enrich subprocess failures with the stderr tail.
                    let detail = self
                        .child
                        .as_ref()
                        .map(|c| {
                            let tail = c.stderr_tail();
                            if tail.is_empty() {
                                message.clone()
                            } else {
                                format!("{message}; stderr tail: {tail}")
                            }
                        })
                        .unwrap_or(message);
                    return Err(FwError::Io(std::io::Error::other(detail)));
                }
                let total_overruns = self.shared.overrun_frames.load(Ordering::Relaxed);
                maybe_log_overruns(total_overruns, &self.shared.overrun_logged, "pipe");
                debug_assert_eq!(written % ch, 0);
                let ended = ended_flag && self.consumer.is_empty();
                if ended && let Some(child) = self.child.as_mut() {
                    child.finish_after_stdout_eof(PIPE_CHILD_EXIT_WAIT)?;
                }
                return Ok(CaptureRead {
                    frames: written / ch,
                    ended,
                    overrun_frames_dropped: total_overruns,
                });
            }
            std::thread::sleep(READ_POLL.min(deadline.saturating_duration_since(Instant::now())));
        }
    }

    fn stop(&mut self) -> FwResult<()> {
        self.shared.ended.store(true, Ordering::Release);
        let cleanup = match self.child.take() {
            Some(mut child) => child.kill().map(drop),
            None => Ok(()),
        };
        let join = self.join.take();
        let reader_cleanup = if cleanup.is_ok() && self.join_reader_on_stop {
            match join {
                Some(join) if join.join().is_err() => Err(FwError::ContractViolation(
                    "pipe capture reader panicked during shutdown".to_owned(),
                )),
                Some(_) | None => Ok(()),
            }
        } else {
            // Dropping JoinHandle detaches the reader. This is required for
            // stdin and is also the only bounded response when process-tree
            // cleanup failed and a descendant may still retain stdout.
            drop(join);
            Ok(())
        };
        cleanup?;
        reader_cleanup
    }
}

impl Drop for PipePcmSource {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            tracing::warn!(%error, "pipe capture cleanup failed during drop");
        }
    }
}

fn decode_sample(format: PcmFormat, bytes: &[u8]) -> f32 {
    match format {
        PcmFormat::S16le => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0,
        PcmFormat::F32le => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

/// Raw PCM on our own stdin (`fw robot listen --source stdin-pcm`): the
/// zero-device-dependency composition path (any upstream producer over any
/// pipe/SSH). Refuses a terminal stdin — an interactive TTY cannot be a PCM
/// stream, and reading it would hang the session confusingly.
pub struct StdinPcmSource;

impl StdinPcmSource {
    /// `stdin_is_terminal` comes from the caller
    /// (`std::io::IsTerminal::is_terminal(&std::io::stdin())`) so the guard
    /// stays unit-testable.
    pub fn open(
        format: PcmFormat,
        sample_rate: u32,
        channels: u16,
        ring_capacity_sec: f64,
        stdin_is_terminal: bool,
    ) -> FwResult<PipePcmSource> {
        if stdin_is_terminal {
            return Err(FwError::InvalidRequest(
                "--source stdin-pcm requires piped raw PCM on stdin; stdin is a terminal \
                 (pipe audio in, e.g. `ffmpeg -i input -f s16le - | fw robot listen --source stdin-pcm`)"
                    .into(),
            ));
        }
        tracing::info!(?format, sample_rate, channels, "stdin PCM capture open");
        PipePcmSource::from_reader(
            std::io::stdin(),
            format,
            sample_rate,
            channels,
            ring_capacity_sec,
        )
    }
}

/// Live device capture through ffmpeg's device layer — the fallback when
/// cpal cannot serve (exotic ALSA setups, containers, explicit
/// `--capture-backend ffmpeg`). ffmpeg downmixes and resamples to 16 kHz
/// mono itself; the low-latency flags matter (without
/// `-fflags nobuffer -flags low_delay` ffmpeg buffers 100-300 ms internally).
pub struct FfmpegCaptureSource;

impl FfmpegCaptureSource {
    pub fn open(
        device: Option<&str>,
        format_override: Option<&str>,
        source_override: Option<&str>,
        ring_capacity_sec: f64,
    ) -> FwResult<PipePcmSource> {
        let (format, source) =
            crate::audio::microphone_defaults(device, format_override, source_override)?;
        let program = crate::audio::resolve_ffmpeg_program(None)?;
        let args: Vec<String> = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            format.as_str(),
            "-i",
            source.as_str(),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        Self::open_with_command(&program, &args, ring_capacity_sec)
    }

    /// Command-injection seam for tests (a fake producer stands in for
    /// ffmpeg; the production caller is [`Self::open`]).
    pub fn open_with_command(
        program: &str,
        args: &[String],
        ring_capacity_sec: f64,
    ) -> FwResult<PipePcmSource> {
        // Validate before spawning: an invalid public API value must not leave
        // an avoidable child-process side effect for cleanup to recover.
        capture_ring_capacity_samples(16_000, 1, ring_capacity_sec)?;
        let mut child = crate::process::spawn_streaming_stdout(program, args)?;
        let stdout = child.take_stdout().ok_or_else(|| {
            FwError::ContractViolation("streaming child spawned without a stdout pipe".into())
        })?;
        tracing::info!(command = %child.rendered_command(), "ffmpeg capture stream open");
        Ok(
            PipePcmSource::from_reader(stdout, PcmFormat::S16le, 16_000, 1, ring_capacity_sec)?
                .with_child(child),
        )
    }
}

// ---------------------------------------------------------------------------
// bd-rt-resampler-pbk9: streaming resampler to 16 kHz mono
//
// Dependency decision (the bead's required evaluation, 2026-08-22): rubato
// v5 drags in an FFT stack (num-complex/realfft path), the audioadapter
// family, windowfunctions, and a proc-macro chain — ~10 transitive crates
// for a task that needs one polyphase FIR. The bead's own caveat ("NO FFT
// requirement at this rate") decides it: hand-rolled kaiser windowed-sinc
// polyphase (upfirdn structure), zero new dependencies. Design strength is
// rubato-class: 70 dB stopband kaiser (beta ~6.76), passband to
// 0.45*out_rate, stopband at the output Nyquist; tap count derived from the
// kaiser formula per ratio (48k->16k: ~260 taps; 44.1k->16k: ~239
// taps/phase across 160 phases). Per-output cost is taps_per_phase MACs —
// single-digit MMAC/s, immaterial.
//
// The batch pipeline's linear-interpolation resampler is deliberately NOT
// touched (byte-exactness-protected, own ledger history); this is new
// surface for the live path only.
// ---------------------------------------------------------------------------

/// Streaming sample-rate converter: device-native interleaved audio in,
/// 16 kHz (or any target) MONO f32 out. Filter state carries across chunks
/// (no per-chunk edge clicks); [`StreamingResampler::flush`] drains the
/// tail at session end.
///
/// Group delay is (taps_per_phase-1)/2 input samples (sub-millisecond at
/// device rates) and is deliberately ignored by the session clock
/// (bd-rt-buffer note: far under the 20 ms mel frame).
pub struct StreamingResampler {
    in_rate: u32,
    out_rate: u32,
    channels: usize,
    /// Upsample / downsample factors, reduced (out/in = l/m).
    l: u64,
    m: u64,
    /// Taps per polyphase branch (K). Filter length is K*L.
    taps_per_phase: usize,
    /// Prototype filter, upsampled-domain, length K*L, gain-compensated.
    filter: Vec<f32>,
    /// Downmixed mono input not yet fully consumed. Prepadded with K-1
    /// zeros at construction so the first outputs have history.
    buf: Vec<f32>,
    /// Absolute input-sample index of `buf[0]` (including the zero prepad,
    /// which occupies indices 0..K-1; real audio starts at K-1).
    buf_start: u64,
    /// Upsampled-domain position of the next output sample (advances by M).
    t: u64,
    /// Carry for a partial interleaved frame split across chunk boundaries.
    pending_frame: Vec<f32>,
    flushed: bool,
}

#[derive(Debug, Clone, Copy)]
struct ResamplerLayout {
    l: u64,
    m: u64,
    taps_per_phase: usize,
    filter_len: usize,
}

impl StreamingResampler {
    fn checked_layout(in_rate: u32, channels: u16, out_rate: u32) -> FwResult<ResamplerLayout> {
        if in_rate == 0 || out_rate == 0 || channels == 0 {
            return Err(FwError::InvalidRequest(
                "resampler requires nonzero rates and channel count".into(),
            ));
        }
        if channels > MAX_RESAMPLER_CHANNELS {
            return Err(FwError::InvalidRequest(format!(
                "resampler channel count {channels} exceeds the {MAX_RESAMPLER_CHANNELS}-channel safety limit"
            )));
        }

        let g = gcd(u64::from(in_rate), u64::from(out_rate));
        let l = u64::from(out_rate) / g;
        let m = u64::from(in_rate) / g;
        let l_usize = usize::try_from(l).map_err(|_| {
            FwError::InvalidRequest("resampler phase count does not fit in usize".to_owned())
        })?;

        // Kaiser design, A = 70 dB stopband, transition from 0.45*out to
        // 0.5*out (all in Hz), normalized to the upsampled rate in*L.
        let fs_up = f64::from(in_rate) * l as f64;
        let attenuation_db = 70.0;
        let transition = 0.05 * f64::from(out_rate) / fs_up;
        let estimated_len = ((attenuation_db - 7.95) / (14.36 * transition)).ceil();
        if !estimated_len.is_finite() || estimated_len > MAX_RESAMPLER_FILTER_TAPS as f64 {
            return Err(FwError::InvalidRequest(format!(
                "resampler filter for {in_rate} Hz -> {out_rate} Hz requires approximately {estimated_len:.0} taps; maximum is {MAX_RESAMPLER_FILTER_TAPS}"
            )));
        }
        let taps_per_phase = (estimated_len as usize).div_ceil(l_usize).max(4);
        if taps_per_phase > MAX_RESAMPLER_TAPS_PER_PHASE {
            return Err(FwError::InvalidRequest(format!(
                "resampler filter for {in_rate} Hz -> {out_rate} Hz requires {taps_per_phase} taps per output phase; maximum is {MAX_RESAMPLER_TAPS_PER_PHASE}"
            )));
        }
        let filter_len = taps_per_phase.checked_mul(l_usize).ok_or_else(|| {
            FwError::InvalidRequest("resampler filter length overflows usize".to_owned())
        })?;
        if filter_len > MAX_RESAMPLER_FILTER_TAPS {
            return Err(FwError::InvalidRequest(format!(
                "resampler filter for {in_rate} Hz -> {out_rate} Hz requires {filter_len} taps after phase alignment; maximum is {MAX_RESAMPLER_FILTER_TAPS}"
            )));
        }
        Ok(ResamplerLayout {
            l,
            m,
            taps_per_phase,
            filter_len,
        })
    }

    /// Validate rate/channel-derived allocation bounds without constructing a
    /// filter. The listen driver uses this before resolving external resources.
    pub(crate) fn validate_config(in_rate: u32, channels: u16, out_rate: u32) -> FwResult<()> {
        Self::checked_layout(in_rate, channels, out_rate).map(|_| ())
    }

    /// Create a resampler from `in_rate` Hz interleaved `channels` audio to
    /// `out_rate` Hz mono.
    pub fn new(in_rate: u32, channels: u16, out_rate: u32) -> FwResult<Self> {
        let layout = Self::checked_layout(in_rate, channels, out_rate)?;
        let l = layout.l;
        let m = layout.m;

        // Kaiser design, A = 70 dB stopband, transition from 0.45*out to
        // 0.5*out (all in Hz), normalized to the upsampled rate in*L.
        let fs_up = f64::from(in_rate) * l as f64;
        let attenuation_db = 70.0;
        let beta = 0.1102 * (attenuation_db - 8.7);
        let taps_per_phase = layout.taps_per_phase;
        let n = layout.filter_len;

        let cutoff = 0.475 * f64::from(out_rate) / fs_up; // cycles/sample, mid-transition
        let center = (n - 1) as f64 / 2.0;
        let i0_beta = bessel_i0(beta);
        let gain = l as f64;
        let mut filter = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64 - center;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
            };
            let w = {
                let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta
            };
            filter.push((gain * sinc * w) as f32);
        }

        let taps = taps_per_phase;
        Ok(Self {
            in_rate,
            out_rate,
            channels: usize::from(channels),
            l,
            m,
            taps_per_phase: taps,
            filter,
            buf: vec![0.0; taps - 1], // history prepad
            buf_start: 0,
            // Absolute index space includes the prepad: real input sample j
            // lives at absolute index (K-1)+j, and output n is centered at
            // real position n*M/L, i.e. absolute base (K-1) + n*M/L. Start t
            // so that base(t=start) = K-1.
            t: (taps as u64 - 1) * l,
            pending_frame: Vec::with_capacity(usize::from(channels)),
            flushed: false,
        })
    }

    #[must_use]
    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    #[must_use]
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    /// Feed interleaved input samples; append produced mono output samples
    /// to `out`. Chunk boundaries may split frames — a partial frame is
    /// carried to the next call.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> FwResult<()> {
        if self.flushed {
            return Err(FwError::InvalidRequest(
                "resampler already flushed; create a new one per session".into(),
            ));
        }
        // Reassemble whole frames across chunk boundaries, then downmix by
        // averaging channels (same rule as the batch mixer).
        let mut samples = input;
        while !self.pending_frame.is_empty() && !samples.is_empty() {
            self.pending_frame.push(samples[0]);
            samples = &samples[1..];
            if self.pending_frame.len() == self.channels {
                let mono = self.pending_frame.iter().sum::<f32>() / self.channels as f32;
                self.buf.push(mono);
                self.pending_frame.clear();
            }
        }
        let whole = samples.len() - samples.len() % self.channels;
        for frame in samples[..whole].chunks_exact(self.channels) {
            self.buf
                .push(frame.iter().sum::<f32>() / self.channels as f32);
        }
        self.pending_frame.extend_from_slice(&samples[whole..]);

        self.emit_ready(out);
        Ok(())
    }

    /// Drain the filter tail at end of session (pads history-length zeros).
    /// The resampler is unusable afterwards.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if self.flushed {
            return;
        }
        // An incomplete trailing frame is dropped (cannot be downmixed).
        self.pending_frame.clear();
        self.buf
            .extend(std::iter::repeat_n(0.0f32, self.taps_per_phase - 1));
        self.emit_ready(out);
        self.flushed = true;
    }

    /// Emit every output sample whose full tap window is buffered.
    fn emit_ready(&mut self, out: &mut Vec<f32>) {
        let k = self.taps_per_phase;
        let buf_end = self.buf_start + self.buf.len() as u64;
        loop {
            let base = self.t / self.l; // newest input index the window touches
            if base >= buf_end {
                break;
            }
            let phase = (self.t % self.l) as usize;
            // Window spans input indices [base-k+1 ..= base]; the prepad
            // guarantees base-k+1 >= buf_start for all reachable t.
            let start = (base + 1 - k as u64 - self.buf_start) as usize;
            let window = &self.buf[start..start + k];
            let mut acc = 0.0f32;
            // filter index for x[base - j] is phase + j*L, j = 0..K;
            // window[k-1-j] = x[base-j].
            for (j, &coeff) in (0..k).map(|j| (j, &self.filter[phase + j * self.l as usize])) {
                acc += coeff * window[k - 1 - j];
            }
            out.push(acc);
            self.t += self.m;
        }
        // Drop input no future output can touch: the oldest sample the next
        // window needs is (t/L) - (K-1).
        let needed_from = (self.t / self.l).saturating_sub(k as u64 - 1);
        if needed_from > self.buf_start {
            let drop = (needed_from - self.buf_start) as usize;
            let drop = drop.min(self.buf.len());
            self.buf.drain(..drop);
            self.buf_start += drop as u64;
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Modified Bessel function of the first kind, order zero (kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    for k in 1..64 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < sum * 1e-14 {
            break;
        }
    }
    sum
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

    fn stop(&mut self) -> FwResult<()> {
        self.ended = true;
        Ok(())
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
        src.stop().expect("stop fixture source");
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

    // -----------------------------------------------------------------------
    // bd-rt-resampler-pbk9: streaming resampler
    // -----------------------------------------------------------------------

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        let n = (f64::from(rate) * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// Goertzel power of `freq` in `x` at `rate` (steady-state region only:
    /// skips the first/last 10% to avoid filter edge transients).
    fn goertzel(x: &[f32], rate: u32, freq: f64) -> f64 {
        let skip = x.len() / 10;
        let x = &x[skip..x.len() - skip];
        let w = 2.0 * std::f64::consts::PI * freq / f64::from(rate);
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &v in x {
            let s0 = f64::from(v) + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        ((s1 * s1 + s2 * s2 - coeff * s1 * s2) / (x.len() as f64 * x.len() as f64 / 4.0)).sqrt()
    }

    fn resample_all(rs: &mut StreamingResampler, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        rs.process(input, &mut out).unwrap();
        rs.flush(&mut out);
        out
    }

    #[test]
    fn resampler_48k_preserves_in_band_tone() {
        let mut rs = StreamingResampler::new(48_000, 1, 16_000).unwrap();
        let out = resample_all(&mut rs, &tone(1_000.0, 48_000, 1.0));
        // Length ~ in/3 (+ small flush tail).
        assert!(
            out.len() >= 16_000 && out.len() < 16_400,
            "len {}",
            out.len()
        );
        // 1 kHz survives at close to unit amplitude.
        let level = goertzel(&out, 16_000, 1_000.0);
        assert!(level > 0.85, "1 kHz level after resample: {level}");
        // Overall energy is tone-like (sine RMS ~ 0.707), not inflated.
        let r = rms(&out);
        assert!((r - 0.707).abs() < 0.05, "rms {r}");
    }

    #[test]
    fn resampler_48k_suppresses_above_nyquist_alias() {
        // 10 kHz is above the 8 kHz output Nyquist: virtually nothing may
        // reach the output (this is THE anti-alias requirement; linear
        // interpolation fails it, which is why this resampler exists).
        let mut rs = StreamingResampler::new(48_000, 1, 16_000).unwrap();
        let out = resample_all(&mut rs, &tone(10_000.0, 48_000, 1.0));
        let steady = &out[out.len() / 10..out.len() * 9 / 10];
        let level = rms(steady);
        // -60 dB relative to the 0.707 input RMS.
        assert!(
            level < 0.707 * 1e-3,
            "alias leakage {level} (-{:.1} dB)",
            -20.0 * (level / 0.707).log10()
        );
    }

    #[test]
    fn resampler_44k1_rational_ratio_preserves_tone() {
        let mut rs = StreamingResampler::new(44_100, 1, 16_000).unwrap();
        let out = resample_all(&mut rs, &tone(1_000.0, 44_100, 1.0));
        let expected = 44_100 / 441 * 160; // in * 160/441
        assert!(
            (out.len() as i64 - i64::from(expected)).unsigned_abs() < 400,
            "len {} vs ~{expected}",
            out.len()
        );
        let level = goertzel(&out, 16_000, 1_000.0);
        assert!(level > 0.85, "1 kHz level after 44.1k resample: {level}");
    }

    #[test]
    fn resampler_stereo_downmix_averages_channels() {
        // L = tone, R = -tone: average is silence.
        let mono = tone(440.0, 48_000, 0.5);
        let interleaved: Vec<f32> = mono.iter().flat_map(|&v| [v, -v]).collect();
        let mut rs = StreamingResampler::new(48_000, 2, 16_000).unwrap();
        let out = resample_all(&mut rs, &interleaved);
        assert!(
            rms(&out) < 1e-6,
            "L/-R downmix must cancel, rms {}",
            rms(&out)
        );

        // L = R = tone: average preserves it.
        let interleaved: Vec<f32> = mono.iter().flat_map(|&v| [v, v]).collect();
        let mut rs = StreamingResampler::new(48_000, 2, 16_000).unwrap();
        let out = resample_all(&mut rs, &interleaved);
        assert!(goertzel(&out, 16_000, 440.0) > 0.85);
    }

    #[test]
    fn resampler_chunked_equals_whole_bitwise() {
        let input = tone(1_234.0, 48_000, 0.7);
        let mut whole = StreamingResampler::new(48_000, 1, 16_000).unwrap();
        let expected = resample_all(&mut whole, &input);

        // Deterministic ragged chunking (incl. odd sizes).
        let mut chunked = StreamingResampler::new(48_000, 1, 16_000).unwrap();
        let mut out = Vec::new();
        let mut pos = 0;
        let mut step = 1;
        while pos < input.len() {
            let end = (pos + step).min(input.len());
            chunked.process(&input[pos..end], &mut out).unwrap();
            pos = end;
            step = step % 977 + 13; // varied chunk sizes
        }
        chunked.flush(&mut out);
        assert_eq!(expected, out, "chunked processing must be bit-identical");
    }

    #[test]
    fn resampler_stereo_chunk_split_mid_frame_is_safe() {
        let mono = tone(500.0, 48_000, 0.2);
        let interleaved: Vec<f32> = mono.iter().flat_map(|&v| [v, v]).collect();
        let mut whole = StreamingResampler::new(48_000, 2, 16_000).unwrap();
        let expected = resample_all(&mut whole, &interleaved);

        let mut split = StreamingResampler::new(48_000, 2, 16_000).unwrap();
        let mut out = Vec::new();
        // Push in chunks of 3 samples: every other chunk splits a frame.
        for chunk in interleaved.chunks(3) {
            split.process(chunk, &mut out).unwrap();
        }
        split.flush(&mut out);
        assert_eq!(expected, out);
    }

    #[test]
    fn resampler_silence_in_silence_out() {
        let mut rs = StreamingResampler::new(48_000, 2, 16_000).unwrap();
        let out = resample_all(&mut rs, &vec![0.0f32; 48_000]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn resampler_rejects_zero_config_and_double_flush_use() {
        assert!(StreamingResampler::new(0, 1, 16_000).is_err());
        assert!(StreamingResampler::new(48_000, 0, 16_000).is_err());
        assert!(StreamingResampler::new(48_000, 1, 0).is_err());
        let excessive_channels =
            StreamingResampler::validate_config(48_000, MAX_RESAMPLER_CHANNELS + 1, 16_000)
                .expect_err("unsafe channel count must fail before allocation");
        assert!(excessive_channels.to_string().contains("channel count"));
        let excessive_filter = StreamingResampler::validate_config(u32::MAX, 1, 16_000)
            .expect_err("unsafe rational filter must fail before allocation");
        assert!(excessive_filter.to_string().contains("filter"));
        let excessive_work = StreamingResampler::validate_config(16_000_000, 1, 16_000)
            .expect_err("unsafe per-output work must fail before allocation");
        assert!(excessive_work.to_string().contains("per output phase"));
        let mut rs = StreamingResampler::new(48_000, 1, 16_000).unwrap();
        let mut out = Vec::new();
        rs.flush(&mut out);
        rs.flush(&mut out); // idempotent
        let err = rs.process(&[0.0], &mut out).unwrap_err();
        assert_eq!(err.error_code(), "FW-INVALID-REQUEST");
    }

    #[test]
    fn resampler_equal_rates_mono_is_near_identity() {
        // 16k -> 16k still runs the (allpass-band) filter; a mid-band tone
        // must survive essentially unchanged.
        let mut rs = StreamingResampler::new(16_000, 1, 16_000).unwrap();
        let out = resample_all(&mut rs, &tone(1_000.0, 16_000, 0.5));
        let level = goertzel(&out, 16_000, 1_000.0);
        assert!(level > 0.9, "identity-ratio level {level}");
    }

    #[test]
    #[ignore = "throughput sanity, host-dependent; run manually"]
    fn resampler_throughput_sanity() {
        let input = tone(1_000.0, 48_000, 1.0);
        let interleaved: Vec<f32> = input.iter().flat_map(|&v| [v, v]).collect();
        let mut rs = StreamingResampler::new(48_000, 2, 16_000).unwrap();
        let started = Instant::now();
        let mut out = Vec::new();
        rs.process(&interleaved, &mut out).unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(5),
            "1 s of 48 kHz stereo took {elapsed:?} (expected << 5 ms)"
        );
    }

    // -----------------------------------------------------------------------
    // bd-rt-ffmpeg-pipe-7dbu: pipe-fed capture sources
    // -----------------------------------------------------------------------

    /// In-memory reader delivering bytes in scripted chunk sizes (exercises
    /// partial-sample and partial-frame reassembly).
    struct ChunkedReader {
        data: Vec<u8>,
        pos: usize,
        sizes: Vec<usize>,
        call: usize,
    }

    impl std::io::Read for ChunkedReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let want = self.sizes[self.call % self.sizes.len()].min(buf.len());
            self.call += 1;
            let n = want.min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    struct BlockingReader {
        entered: std::sync::mpsc::SyncSender<()>,
        release: std::sync::mpsc::Receiver<()>,
    }

    impl std::io::Read for BlockingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            let _ = self.entered.try_send(());
            let _ = self.release.recv();
            Ok(0)
        }
    }

    fn drain_source(src: &mut PipePcmSource) -> Vec<f32> {
        let mut out = vec![0f32; 4096];
        let mut got = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let r = src.read(&mut out, Duration::from_millis(50)).unwrap();
            got.extend_from_slice(&out[..r.frames * usize::from(src.channels())]);
            if r.ended {
                return got;
            }
            assert!(Instant::now() < deadline, "pipe drain hung");
        }
    }

    #[test]
    fn pipe_s16le_round_trips_edge_values_bit_exactly() {
        let values: Vec<i16> = vec![i16::MIN, -1, 0, 1, i16::MAX, 12345, -12345, 32767];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Chunk sizes deliberately split samples across reads (1 and 3 bytes).
        let reader = ChunkedReader {
            data: bytes,
            pos: 0,
            sizes: vec![1, 3, 2, 5],
            call: 0,
        };
        let mut src = PipePcmSource::from_reader(reader, PcmFormat::S16le, 16_000, 1, 1.0).unwrap();
        let got = drain_source(&mut src);
        let expected: Vec<f32> = values.iter().map(|&v| f32::from(v) / 32_768.0).collect();
        assert_eq!(got, expected, "s16le conversion must be bit-exact");
    }

    #[test]
    fn pipe_f32le_passes_values_verbatim_across_ragged_chunks() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.25, f32::MIN_POSITIVE];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let reader = ChunkedReader {
            data: bytes,
            pos: 0,
            sizes: vec![3, 1, 7],
            call: 0,
        };
        let mut src = PipePcmSource::from_reader(reader, PcmFormat::F32le, 48_000, 1, 1.0).unwrap();
        assert_eq!(drain_source(&mut src), values);
    }

    #[test]
    fn pipe_stereo_frame_alignment_survives_odd_chunks() {
        // 100 stereo frames, delivered in 3-byte chunks (splits both samples
        // and frames); output must be exactly the interleaved input.
        let values: Vec<i16> = (0..200).map(|i| i as i16).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let reader = ChunkedReader {
            data: bytes,
            pos: 0,
            sizes: vec![3],
            call: 0,
        };
        let mut src = PipePcmSource::from_reader(reader, PcmFormat::S16le, 16_000, 2, 1.0).unwrap();
        let got = drain_source(&mut src);
        assert_eq!(got.len(), 200);
        let expected: Vec<f32> = values.iter().map(|&v| f32::from(v) / 32_768.0).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn pipe_rejects_truncated_samples_and_frames_at_eof() {
        let cases = [
            ("partial s16le sample", PcmFormat::S16le, 1, vec![0]),
            (
                "partial f32le sample",
                PcmFormat::F32le,
                1,
                vec![0, 0, 0],
            ),
            (
                "partial stereo frame",
                PcmFormat::S16le,
                2,
                1_i16.to_le_bytes().to_vec(),
            ),
        ];

        for (name, format, channels, bytes) in cases {
            let trailing_bytes = bytes.len();
            let reader = ChunkedReader {
                data: bytes,
                pos: 0,
                sizes: vec![1],
                call: 0,
            };
            let mut source =
                PipePcmSource::from_reader(reader, format, 16_000, channels, 1.0).unwrap();
            let mut out = [0.0f32; 4];
            let deadline = Instant::now() + Duration::from_secs(5);
            let error = loop {
                match source.read(&mut out, Duration::from_millis(50)) {
                    Ok(read) => assert!(
                        !read.ended,
                        "{name} was silently accepted as clean end-of-stream"
                    ),
                    Err(error) => break error,
                }
                assert!(Instant::now() < deadline, "{name} did not surface an error");
            };
            assert_eq!(error.error_code(), "FW-IO");
            let message = error.to_string();
            assert!(message.contains("incomplete PCM frame"), "{name}: {message}");
            assert!(
                message.contains(&format!("{trailing_bytes} trailing byte")),
                "{name}: {message}"
            );
        }
    }

    #[test]
    fn pcm_format_parse_accepts_exactly_two_formats() {
        assert_eq!(PcmFormat::parse("s16le").unwrap(), PcmFormat::S16le);
        assert_eq!(PcmFormat::parse("f32le").unwrap(), PcmFormat::F32le);
        let err = PcmFormat::parse("mulaw").unwrap_err();
        assert_eq!(err.error_code(), "FW-INVALID-REQUEST");
        assert!(err.to_string().contains("s16le, f32le"));
    }

    #[test]
    fn stdin_source_refuses_terminal_stdin() {
        let err = StdinPcmSource::open(PcmFormat::S16le, 16_000, 1, 1.0, true).unwrap_err();
        assert_eq!(err.error_code(), "FW-INVALID-REQUEST");
        assert!(err.to_string().contains("terminal"));
    }

    #[test]
    fn command_capture_rejects_unsafe_ring_before_spawning() {
        let error = FfmpegCaptureSource::open_with_command(
            "definitely-missing-capture-producer",
            &[],
            f64::INFINITY,
        )
        .expect_err("invalid ring must win over missing program resolution");
        assert_eq!(error.error_code(), "FW-INVALID-REQUEST");
        assert!(error.to_string().contains("capture ring capacity"));
    }

    #[test]
    fn pipe_zero_config_rejected() {
        let reader = ChunkedReader {
            data: vec![],
            pos: 0,
            sizes: vec![1],
            call: 0,
        };
        assert!(PipePcmSource::from_reader(reader, PcmFormat::S16le, 0, 1, 1.0).is_err());
        let reader = ChunkedReader {
            data: vec![],
            pos: 0,
            sizes: vec![1],
            call: 0,
        };
        assert!(PipePcmSource::from_reader(reader, PcmFormat::S16le, 16_000, 0, 1.0).is_err());
    }

    #[test]
    fn capture_ring_capacity_is_finite_positive_and_memory_bounded() {
        assert_eq!(
            capture_ring_capacity_samples(48_000, 2, 300.0).expect("five-minute stereo ring"),
            28_800_000
        );
        for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = capture_ring_capacity_samples(16_000, 1, seconds)
                .expect_err("invalid ring duration must fail closed");
            assert_eq!(error.error_code(), "FW-INVALID-REQUEST");
        }
        let error = capture_ring_capacity_samples(u32::MAX, u16::MAX, 300.0)
            .expect_err("catastrophic ring request must fail before allocation");
        assert_eq!(error.error_code(), "FW-INVALID-REQUEST");
        assert!(error.to_string().contains("maximum"));
    }

    #[test]
    fn externally_owned_blocking_reader_stop_detaches_promptly() {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let reader = BlockingReader {
            entered: entered_tx,
            release: release_rx,
        };
        let mut src = PipePcmSource::from_reader(reader, PcmFormat::S16le, 16_000, 1, 1.0).unwrap();
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("reader entered its externally controlled blocking read");

        let started = Instant::now();
        src.stop().expect("detach externally owned reader");
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "stop waited for externally owned input: {:?}",
            started.elapsed()
        );
        drop(release_tx);
    }

    // ---- subprocess-backed source (unix-only helpers; sh is universal) ----

    #[cfg(not(windows))]
    #[test]
    fn ffmpeg_command_source_streams_incrementally_before_child_exit() {
        // Fake producer: emits 1600 samples of s16le, sleeps long, then would
        // emit more — incremental delivery must hand us the first batch while
        // the child still lives, and stop() must kill + reap without hanging.
        let script = "dd if=/dev/zero bs=3200 count=1 2>/dev/null; sleep 30";
        let args: Vec<String> = vec!["-c".into(), script.into()];
        let mut src = FfmpegCaptureSource::open_with_command("sh", &args, 1.0).unwrap();
        let mut out = vec![0f32; 4096];
        let started = Instant::now();
        let mut frames = 0usize;
        while frames == 0 && started.elapsed() < Duration::from_secs(5) {
            frames = src
                .read(&mut out, Duration::from_millis(100))
                .unwrap()
                .frames;
        }
        assert_eq!(frames, 1600, "first batch must arrive while the child runs");
        let t_stop = Instant::now();
        src.stop().expect("stop subprocess capture");
        assert!(
            t_stop.elapsed() < Duration::from_secs(2),
            "stop() must kill promptly, took {:?}",
            t_stop.elapsed()
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn ffmpeg_command_source_surfaces_stderr_tail_on_failure() {
        let args: Vec<String> = vec!["-c".into(), "echo device gone >&2; exit 3".into()];
        let mut src = FfmpegCaptureSource::open_with_command("sh", &args, 1.0).unwrap();
        let mut out = vec![0f32; 64];
        let deadline = Instant::now() + Duration::from_secs(5);
        let error = loop {
            match src.read(&mut out, Duration::from_millis(50)) {
                Ok(read) => assert!(
                    !read.ended,
                    "nonzero producer exit was converted to successful EOF"
                ),
                Err(error) => break error,
            }
            assert!(
                Instant::now() < deadline,
                "failed child did not surface its exit status"
            );
        };
        assert_eq!(error.error_code(), "FW-CMD-FAILED");
        let text = error.to_string();
        assert!(text.contains("status: 3"), "exit status was lost: {text}");
        assert!(text.contains("device gone"), "stderr tail was lost: {text}");
    }

    #[cfg(not(windows))]
    #[test]
    fn streaming_child_kill_reaps_no_zombie() {
        let args: Vec<String> = vec!["-c".into(), "sleep 30".into()];
        let mut child = crate::process::spawn_streaming_stdout("sh", &args).unwrap();
        let status = child.kill().unwrap();
        assert!(status.is_some(), "kill must reap and return the status");
        // Second kill is a no-op.
        assert!(child.kill().unwrap().is_none());
    }

    #[cfg(not(windows))]
    #[test]
    fn streaming_child_missing_program_maps_to_command_missing() {
        let err = crate::process::spawn_streaming_stdout("definitely-not-a-real-binary-fw", &[])
            .unwrap_err();
        assert_eq!(err.error_code(), "FW-CMD-MISSING");
    }

    // -----------------------------------------------------------------------
    // bd-rt-device-probe-wh02: watchdog + taxonomy
    // -----------------------------------------------------------------------

    #[test]
    fn silent_watchdog_fires_exactly_once_at_threshold() {
        let mut watchdog = SilentInputWatchdog::new(1.0, 16_000);
        // 0.9 s of zeros: below threshold.
        assert!(!watchdog.observe(&vec![0.0; 14_400]));
        // Crossing 1.0 s: fires exactly once.
        assert!(watchdog.observe(&vec![0.0; 1_600]));
        // Never again.
        assert!(!watchdog.observe(&vec![0.0; 32_000]));
    }

    #[test]
    fn silent_watchdog_disarmed_by_any_nonzero_sample() {
        let mut watchdog = SilentInputWatchdog::new(1.0, 16_000);
        let mut quiet = vec![0.0f32; 8_000];
        quiet.push(1e-9); // even sub-audible dither disarms
        assert!(!watchdog.observe(&quiet));
        assert!(
            !watchdog.observe(&vec![0.0; 160_000]),
            "disarm is permanent"
        );
    }

    #[test]
    fn silent_watchdog_low_level_noise_never_triggers() {
        let mut watchdog = SilentInputWatchdog::new(0.5, 16_000);
        // -60 dBFS constant tone: nonzero from the first sample.
        let noise = vec![0.001f32; 16_000];
        assert!(!watchdog.observe(&noise));
    }

    #[test]
    fn device_not_found_error_names_request_and_inventory() {
        let err = device_not_found_error("Imaginary USB Mic 9000");
        assert_eq!(err.error_code(), "FW-INVALID-REQUEST");
        let message = err.to_string();
        assert!(message.contains("Imaginary USB Mic 9000"));
        assert!(message.contains("available input devices"), "{message}");
    }

    #[test]
    fn enumerate_input_devices_is_deterministic_and_promptless() {
        // Runs on any host (zero devices is a valid outcome, e.g. CI).
        // Asserts the ordering contract and that two enumerations agree —
        // and by running in CI at all, that enumeration never blocks on a
        // permission prompt.
        let Ok(a) = enumerate_input_devices() else {
            eprintln!("SKIP enumerate: host audio backend unavailable");
            return;
        };
        let b = enumerate_input_devices().expect("second enumeration");
        assert_eq!(a, b, "enumeration must be stable");
        if a.len() > 1 {
            assert!(
                a[0].is_default || a.iter().all(|d| !d.is_default),
                "default device must sort first when present"
            );
            for pair in a.windows(2) {
                if !pair[0].is_default && !pair[1].is_default {
                    assert!(pair[0].name <= pair[1].name, "name ordering");
                }
            }
        }
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
        src.stop().expect("stop cpal source");
        assert!(total > 0, "no audio frames captured from live device");
    }
}
