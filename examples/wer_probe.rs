//! Cross-engine WER between two transcript files, using the same
//! [`franken_whisper::conformance::word_error_rate`] the vs-incumbent harness
//! gates on.
//!
//! Text comparison is contention-immune, so this certifies a non-byte-exact
//! encoder change (ToMe token merging) without needing an exclusive host —
//! only the timing cell does.
//!
//! ```text
//! wer_probe <reference.txt> <hypothesis.txt>
//! ```

use franken_whisper::conformance::word_error_rate;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: wer_probe <reference.txt> <hypothesis.txt>");
        std::process::exit(2);
    }
    let reference = std::fs::read_to_string(&args[1]).expect("read reference");
    let hypothesis = std::fs::read_to_string(&args[2]).expect("read hypothesis");
    let report = word_error_rate(&reference, &hypothesis);
    println!(
        "WER wer={:.6} edits={} ref_words={} hyp_words={}",
        report.wer, report.edits, report.reference_words, report.hypothesis_words
    );
}
