// Scratch probe (bd-3nw3): builtin symphonia normalize vs forced-ffmpeg
// subprocess normalize, on the same compressed input. One timed run per
// process; the variant is chosen via FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE
// from the invoking shell (env is read per call, not cached).
use std::path::Path;
use std::time::Instant;

fn main() {
    let input = Path::new("/data/tmp/bd3nw3_600s.mp3");
    let work = Path::new("/data/tmp/bd3nw3_work");
    std::fs::create_dir_all(work).expect("work dir");

    // Warm the page cache so both variants see the same file state.
    let _ = std::fs::metadata(input).expect("input missing");

    let t = Instant::now();
    let out =
        franken_whisper::audio::normalize_to_wav(input, work).expect("normalize must succeed");
    let elapsed = t.elapsed();

    let variant = if std::env::var("FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE").is_ok_and(|v| v == "1")
    {
        "ffmpeg"
    } else {
        "builtin"
    };
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "BD3NW3>>> variant={variant} elapsed_ms={} out={} out_bytes={size}",
        elapsed.as_millis(),
        out.display()
    );

    // Clean the output so the next variant starts from identical disk state.
    let _ = std::fs::remove_file(&out);
}
