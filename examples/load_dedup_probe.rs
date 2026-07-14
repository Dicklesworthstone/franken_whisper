//! Concurrent cold-load dedup probe (FW_LOAD_DEDUP).
//!
//! Spawns `N` threads that all load the SAME model from a cold cache at the same
//! instant (a `Barrier` so none has published yet). Without dedup every thread
//! parses the ~1.5 GB blob (N× parse work + N× peak RSS, oversubscribing the box);
//! with `FW_LOAD_DEDUP=1` one parses and the rest wait then hit the cache.
//!
//! Usage: `N=4 FW_LOAD_DEDUP=1 load_dedup_probe [model-short-name]`
//! (needs `FRANKEN_WHISPER_MODEL_DIR`).

use std::sync::{Arc, Barrier};
use std::time::Instant;

use franken_whisper::native_engine::{NativeWhisperModel, find_model_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "large-v3-turbo".to_string());
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;
    let dedup = std::env::var("FW_LOAD_DEDUP").as_deref() == Ok("1");

    let barrier = Arc::new(Barrier::new(n));
    let t = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let p = path.clone();
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                b.wait(); // all threads start the load at the same instant
                let s = Instant::now();
                let m = NativeWhisperModel::load(&p).expect("model load failed");
                let ms = s.elapsed().as_secs_f64() * 1e3;
                // Touch it so the load isn't optimized away; keep alive until timed.
                let _ = m.loaded();
                ms
            })
        })
        .collect();
    let per: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let total = t.elapsed().as_secs_f64() * 1e3;
    let per_r: Vec<i64> = per.iter().map(|x| x.round() as i64).collect();
    eprintln!("N={n} FW_LOAD_DEDUP={dedup} total={total:.1}ms per_thread_ms={per_r:?}");
    Ok(())
}
