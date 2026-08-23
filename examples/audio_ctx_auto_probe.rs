//! bd-rt-audio-ctx-auto-empty-4c2i repro probe: AudioCtxPolicy::Auto vs
//! Full on short utterance slices (the listen driver's workload). Auto
//! reportedly decodes ZERO tokens on real speech; Full is correct.
use franken_whisper::native_engine::decode::{
    AudioCtxPolicy, DecodeParams, LoadedModel, transcribe_samples,
};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn main() {
    let path = find_model_file("tiny.en").expect("tiny.en cached");
    let model = LoadedModel::from_ggml(GgmlModel::load(&path).expect("load")).expect("prep");
    let mut reader = hound::WavReader::open("tests/fixtures/native/jfk.wav").expect("jfk");
    let all: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("pcm")) / 32768.0)
        .collect();
    let noop = || Ok(());
    for slice_sec in [2.2_f64, 3.0, 4.42, 11.0] {
        let samples = &all[..((16000.0 * slice_sec) as usize).min(all.len())];
        for policy in [AudioCtxPolicy::Full, AudioCtxPolicy::Auto] {
            let params = DecodeParams {
                language: Some("en".to_owned()),
                timestamps: true,
                audio_ctx: policy,
                bypass_transcript_cache: true,
                n_threads: 4,
                ..DecodeParams::default()
            };
            let started = std::time::Instant::now();
            match transcribe_samples(&model, samples, &params, &noop) {
                Ok(out) => {
                    let text: String = out
                        .segments
                        .iter()
                        .map(|s| s.text.trim())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let tokens: usize = out.windows.iter().map(|w| w.tokens).sum();
                    let quality: Vec<String> = out
                        .windows
                        .iter()
                        .map(|w| {
                            format!(
                                "off={} ns={:.3} lp={:.3}",
                                w.window_offset_sec, w.no_speech_prob, w.avg_logprob
                            )
                        })
                        .collect();
                    println!(
                        "slice={slice_sec:5}s policy={policy:?} ms={:5} tokens={tokens:3} attempts={} windows=[{}] dropped={} text={text:?}",
                        started.elapsed().as_millis(),
                        out.work.window_attempts,
                        quality.join(" | "),
                        out.dropped_windows.len(),
                    );
                }
                Err(e) => println!("slice={slice_sec:5}s policy={policy:?} ERROR {e}"),
            }
        }
    }
}
