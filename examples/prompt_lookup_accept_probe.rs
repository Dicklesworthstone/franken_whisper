//! Prompt-lookup / n-gram self-speculation accept-rate simulator for ASR decode
//! (BlackThrush, 2026-07-03).
//!
//! Draft/speculative decoding's realized speedup = R(K) (amortization ceiling, already
//! measured depth-invariant, [[project_draft_decoding_amortization]]) × ACCEPT RATE. The
//! accept rate needs a draft; the cheapest DRAFT-MODEL-FREE draft is prompt-lookup decoding
//! (PLD): at each step, find the most recent prior occurrence of the trailing n-gram in the
//! sequence-so-far and speculate the K tokens that followed it; the target verifies them in
//! ONE forward pass (BYTE-EXACT — accepted tokens are exactly greedy). PLD pays off ONLY when
//! the output repeats n-grams (code, RAG, summarization). ASR output is largely novel, so the
//! hypothesis is a ~0 hit rate — this MEASURES it on real turbo token streams instead of
//! reasoning. `tokens_per_pass` = decode speedup (len / #forward-passes); 1.00 = PLD useless.
//!
//! Input: `TOKENS>>>id,id,...<<<` lines (from `e2e_probe` with `PROBE_DUMP_TOKENS=1`) on stdin.
//! Usage: `... | prompt_lookup_accept_probe [ngram=3] [maxspec=8]`.
use std::io::Read;

/// Simulate PLD over a known greedy token stream. Returns (passes, len, hits).
/// At step i (tokens[..i] produced), match the last `n` tokens against an earlier
/// occurrence; speculate up to `k` following tokens; accept the matching prefix.
/// A forward pass confirms 1 (the normal next token) + accepted speculated tokens.
fn simulate(tokens: &[i32], n: usize, k: usize) -> (usize, usize, usize) {
    let len = tokens.len();
    if len == 0 {
        return (0, 0, 0);
    }
    let (mut i, mut passes, mut hits) = (0usize, 0usize, 0usize);
    while i < len {
        passes += 1;
        let mut accepted = 0usize;
        if i >= n {
            let needle = &tokens[i - n..i];
            // most recent earlier occurrence of the n-gram (search j from i-1 down)
            let mut found: Option<usize> = None;
            let mut j = i.saturating_sub(1);
            while j >= n {
                if &tokens[j - n..j] == needle {
                    found = Some(j);
                    break;
                }
                if j == n {
                    break;
                }
                j -= 1;
            }
            if let Some(j) = found {
                // speculate tokens[j..j+k]; accept matching prefix vs tokens[i..]
                for s in 0..k {
                    if j + s < len && i + s < len && tokens[j + s] == tokens[i + s] {
                        accepted += 1;
                    } else {
                        break;
                    }
                }
                if accepted > 0 {
                    hits += 1;
                }
            }
        }
        i += 1 + accepted;
    }
    (passes, len, hits)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let k: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");

    let mut streams: Vec<Vec<i32>> = Vec::new();
    for line in input.lines() {
        if let (Some(a), Some(b)) = (line.find("TOKENS>>>"), line.find("<<<")) {
            let body = &line[a + 9..b];
            let toks: Vec<i32> = body
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !toks.is_empty() {
                streams.push(toks);
            }
        }
    }
    if streams.is_empty() {
        eprintln!("no TOKENS>>>...<<< lines on stdin");
        return;
    }
    println!(
        "=== prompt-lookup (n-gram) self-speculation accept rate — ASR decode (ngram={n}, maxspec={k}) ==="
    );
    let (mut tot_pass, mut tot_len, mut tot_hits, mut tot_steps) = (0usize, 0usize, 0usize, 0usize);
    for (idx, s) in streams.iter().enumerate() {
        let (passes, len, hits) = simulate(s, n, k);
        let tpp = len as f64 / passes.max(1) as f64;
        println!(
            "  stream {idx}: {len:>4} tok → {passes:>4} passes  tokens/pass={tpp:.3}  spec-hit steps={hits}/{passes}",
        );
        tot_pass += passes;
        tot_len += len;
        tot_hits += hits;
        tot_steps += passes;
    }
    let tpp = tot_len as f64 / tot_pass.max(1) as f64;
    println!("  ─────");
    println!(
        "  TOTAL: {tot_len} tok → {tot_pass} passes  tokens/pass={tpp:.3}  ({:.1}% steps hit a spec)",
        100.0 * tot_hits as f64 / tot_steps.max(1) as f64
    );
    println!(
        "  ⇒ PLD decode speedup = {tpp:.3}× [{}]",
        if tpp >= 1.15 {
            "VIABLE — free byte-exact decode amortization"
        } else {
            "DEAD for ASR — output too novel; needs a real draft model"
        }
    );
}
