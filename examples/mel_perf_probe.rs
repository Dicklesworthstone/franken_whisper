//! Contention-immune perf probe for the mel frontend.
//!
//! Runs `mel::log_mel` a FIXED number of times so that
//! `perf stat -e instructions:u,cycles:u` yields a deterministic *per-iteration*
//! instruction count — independent of shared-box load. This is the instrument
//! used to A/B the `FW_FFT_ZEROINIT` uninit-FFT-scratch lever when wall-clock
//! Criterion benching is swamped by frankentorch-swarm contention (the box's
//! zeroinit-vs-zeroinit noise floor reaches ±5% on the short sparse bench).
//!
//! Usage: `mel_perf_probe [iters] [realistic|dense]`  (defaults: 300 realistic).
//! A/B:   `FW_FFT_ZEROINIT=1 perf stat -e instructions:u,cycles:u mel_perf_probe 300`
//!        vs the same without the env var (the uninit default).
use franken_whisper::native_engine::MelFilterbank;
use franken_whisper::native_engine::mel::{self, N_FREQ_BINS, N_SAMPLES_30S, SAMPLE_RATE};

fn dense_filterbank(n_mel: usize) -> MelFilterbank {
    // Non-negative deterministic weights; the FFT dominates and is value-
    // independent, and the dense projection touches every bin regardless.
    let data: Vec<f32> = (0..n_mel * N_FREQ_BINS)
        .map(|i| 0.25 + 0.5 * ((i % 97) as f32 / 97.0))
        .collect();
    MelFilterbank {
        n_mel,
        n_fft_bins: N_FREQ_BINS,
        data,
    }
}

fn realistic_filterbank(n_mel: usize) -> MelFilterbank {
    // Slaney-style sparse triangles, ~5 nonzero bins/filter — the production case.
    let sr = SAMPLE_RATE as f64;
    let hz_to_mel = |h: f64| 2595.0 * (1.0 + h / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let edges: Vec<f64> = (0..n_mel + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * (i as f64) / ((n_mel + 1) as f64)))
        .collect();
    let bin_hz = |k: usize| (k as f64) * (sr / 2.0) / ((N_FREQ_BINS - 1) as f64);
    let mut data = vec![0.0f32; n_mel * N_FREQ_BINS];
    for m in 0..n_mel {
        let (lo, ce, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        for k in 0..N_FREQ_BINS {
            let f = bin_hz(k);
            let w = if f >= lo && f <= ce && ce > lo {
                (f - lo) / (ce - lo)
            } else if f > ce && f <= hi && hi > ce {
                (hi - f) / (hi - ce)
            } else {
                0.0
            };
            data[m * N_FREQ_BINS + k] = w as f32;
        }
    }
    MelFilterbank {
        n_mel,
        n_fft_bins: N_FREQ_BINS,
        data,
    }
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let sparse = std::env::args()
        .nth(2)
        .map(|s| s != "dense")
        .unwrap_or(true);

    let sr = SAMPLE_RATE as f32;
    let audio: Vec<f32> = (0..N_SAMPLES_30S)
        .map(|i| {
            let t = i as f32 / sr;
            0.9 * (0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 660.0 * t).sin())
        })
        .collect();
    let filters = if sparse {
        realistic_filterbank(80)
    } else {
        dense_filterbank(80)
    };

    let mut acc = 0usize;
    for _ in 0..iters {
        let m = mel::log_mel(&audio, &filters, 8).expect("log_mel");
        acc = acc.wrapping_add(m.n_frames);
    }
    std::hint::black_box(acc);
    eprintln!(
        "mel_perf_probe: iters={iters} {} acc={acc}",
        if sparse { "realistic" } else { "dense" }
    );
}
