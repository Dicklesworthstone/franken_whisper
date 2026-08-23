//! Does `record_token_attn` change decode OUTPUT? It must not: the tap is
//! documented as observation-only ("out_h and the transcript are
//! BIT-IDENTICAL", decoder.rs cross_attention). The listen latency harness
//! (bd-rt-latency-harness-3dkh first campaign) observed different
//! transcripts between arms whose only decode-param difference is this
//! flag. This probe isolates it: same model, same samples, same params,
//! record on vs off, transcripts diffed.
use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel, transcribe_samples};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn main() {
    let path = find_model_file("tiny.en").expect("tiny.en in model cache");
    let model = LoadedModel::from_ggml(GgmlModel::load(&path).expect("load")).expect("prepare");
    let mut reader = hound::WavReader::open("tests/fixtures/native/jfk.wav").expect("open jfk");
    let all: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("pcm")) / 32768.0)
        .collect();
    // The harness's `short` fixture slice, where the divergence showed.
    let samples = &all[..(16000.0 * 6.9) as usize];
    let noop = || Ok(());
    for round in 0..2 {
        let mut texts = Vec::new();
        for record in [false, true] {
            let params = DecodeParams {
                language: Some("en".to_owned()),
                record_token_attn: record,
                bypass_transcript_cache: true,
                n_threads: 4,
                ..DecodeParams::default()
            };
            let out = transcribe_samples(&model, samples, &params, &noop).expect("decode");
            let text: String = out
                .segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let attn_counts: Vec<usize> = out.windows.iter().map(|w| w.token_attn.len()).collect();
            println!("round={round} record={record} attn_counts={attn_counts:?} text={text:?}");
            texts.push(text);
        }
        if texts[0] == texts[1] {
            println!("round={round}: IDENTICAL");
        } else {
            println!("round={round}: DIVERGENT  <-- tap is not observation-only");
        }
    }
}
