// Scratch probe (bd-cy9u): end-to-end TTY transport throughput.
// Times the full encode path (ffmpeg mulaw transcode + zlib + base64 +
// crc32/sha256 + JSON framing) and the full decode path (parse + verify +
// decompress + reassemble) over a synthetic fixture, then reports
// realtime factors and the wire-bandwidth requirement.
use std::io::Cursor;
use std::time::Instant;

fn main() {
    let dir = std::env::temp_dir().join("bdcy9u_probe");
    std::fs::create_dir_all(&dir).expect("work dir");
    let wav = dir.join("fixture_60s.wav");

    // 60 s of 8 kHz mono s16 noise (worst case for zlib: incompressible).
    if !wav.exists() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
        let mut state = 0xDEADBEEFCAFEBABE_u64;
        for _ in 0..60 * 8_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            writer
                .write_sample(((state % 65_536) as i32 - 32_768) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
    }
    let wav_meta = std::fs::metadata(&wav).expect("fixture");
    let audio_seconds = 60.0;

    // ---- ENCODE (full public path; includes the ffmpeg transcode head) ----
    let encoded_path = dir.join("encoded.ndjson");
    let t = Instant::now();
    franken_whisper::tty_audio::encode_to_writer(
        &wav,
        100,
        &mut std::fs::File::create(&encoded_path).expect("enc"),
    )
    .expect("encode");
    let enc = t.elapsed();
    let wire_bytes = std::fs::metadata(&encoded_path).expect("wire").len();

    // ---- DECODE (full public path) ----
    let wire = std::fs::read(&encoded_path).expect("wire read");
    let mut reader = Cursor::new(&wire);
    let t = Instant::now();
    let (_report, raw) =
        franken_whisper::tty_audio::decode_frames_to_raw(&mut reader).expect("decode");
    let dec = t.elapsed();

    println!(
        "BD3NW3CY9U>>> audio_s={audio_seconds} wav_bytes={} wire_bytes={wire_bytes}",
        wav_meta.len()
    );
    println!(
        "BD3NW3CY9U>>> encode_ms={:?} decode_ms={:?}",
        enc.as_millis(),
        dec.as_millis()
    );
    println!(
        "BD3NW3CY9U>>> encode_rt_factor={:.1}x decode_rt_factor={:.1}x (audio seconds processed per wall second)",
        audio_seconds / enc.as_secs_f64(),
        audio_seconds / dec.as_secs_f64()
    );
    println!(
        "BD3NW3CY9U>>> wire_bandwidth_required={:.0} B/s ({:.1} kbit/s)",
        wire_bytes as f64 / audio_seconds,
        wire_bytes as f64 * 8.0 / audio_seconds / 1000.0
    );
    println!("BD3NW3CY9U>>> decoded_bytes={} (mulaw payload)", raw.len());
    let _ = std::fs::remove_dir_all(&dir);
}
