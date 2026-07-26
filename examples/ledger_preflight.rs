//! Candidate preflight — grep the ledger *before* you touch source.
//!
//! Fleet campaign `perf-campaign-20260725`, Meta-Lever #1, broadcast 2.
//! Modelled on frankensqlite's `sql_pipeline_candidate_preflight` (exit 2 =
//! BLOCKED), the mechanism credited with holding that repo at a **1.7%** void
//! rate while every repo that audited once and stopped drifted to 25–91%.
//!
//! ## What it answers
//!
//! "Has this lever already been tried, and if so, is that rejection binding?"
//! Those are two different questions, and conflating them is expensive in both
//! directions — this repo's own ledger records agents re-deriving already-closed
//! levers, *and* records genuinely-live levers being treated as closed because a
//! void row said no.
//!
//! - **BLOCKED (exit 2)** — a prior REJECT row matches *and* records why it was
//!   decidable (A/A null, counted mechanism, accuracy refutation, or a
//!   large-magnitude loss). That rejection stands. Do not re-derive it.
//! - **VOID PRIOR (exit 0)** — a prior REJECT row matches but records none of
//!   those, so it could not distinguish the lever from the harness. It is *not*
//!   binding. Proceed — and record an A/A null this time.
//! - **CLEAR (exit 0)** — no prior row matches.
//!
//! ## Usage
//!
//! ```text
//! cargo run --example ledger_preflight -- <term> [more terms...]
//! cargo run --example ledger_preflight -- sdpa tile
//! ```
//!
//! All terms must appear in the row (AND), case-insensitive.
//!
//! The evidence markers here intentionally mirror `tests/ledger_integrity.rs`,
//! which enforces the same rule on *new* rows. This tool advises before the
//! work; that test blocks the bad row afterwards.

use std::path::Path;

const EVIDENCE_MARKERS: &[&str] = &[
    "null control",
    "a/a",
    "null median",
    "null_p90",
    "null p90",
    "null p10",
    "identity null",
    "base/base",
    "null floor",
    "null pair",
    "instructions",
    "cycles",
    "syscall",
    "allocation",
    "page fault",
    "perf stat",
    "retired",
    "wer",
    "accuracy",
    "faithful",
    "not safe",
    "byte-exact",
    "byte-identical",
    "non-byte-exact",
    "slower",
    "self-time",
    "self time",
    "amdahl",
];

/// Verdict words this repo actually uses to close a lever. Matching only
/// `REJECT` misses half the population: the `int4 mlp_0` family is closed under
/// *DEAD* / *CLOSED* / *FALSIFIED* / *NEGATIVE* and never says "REJECT", so a
/// narrower preflight reports CLEAR on a genuinely dead lever — the expensive
/// direction of the error. Kept in sync with `tests/ledger_integrity.rs`.
const REJECTION_VERDICTS: &[&str] = &[
    "REJECT",
    "DEAD",
    "CLOSED",
    "FALSIFIED",
    "NO-SHIP",
    "DO-NOT-RETRY",
    "NEGATIVE",
];

struct Row {
    line: usize,
    header: String,
    body: String,
}

fn main() {
    let terms: Vec<String> = std::env::args().skip(1).map(|a| a.to_lowercase()).collect();
    if terms.is_empty() {
        eprintln!(
            "usage: cargo run --example ledger_preflight -- <term> [more terms...]\n\
             all terms must appear in a row (AND), case-insensitive"
        );
        std::process::exit(64);
    }

    let ledger = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/NEGATIVE_EVIDENCE.md");
    let text = match std::fs::read_to_string(&ledger) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read {}: {error}", ledger.display());
            std::process::exit(70);
        }
    };

    let mut rows: Vec<Row> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        if let Some(_rest) = raw.strip_prefix("## ") {
            rows.push(Row {
                line: idx + 1,
                header: raw.to_owned(),
                body: String::new(),
            });
        } else if let Some(current) = rows.last_mut() {
            current.body.push_str(raw);
            current.body.push('\n');
        }
    }

    let mut binding = Vec::new();
    let mut void_prior = Vec::new();

    for row in &rows {
        let upper = row.header.to_uppercase();
        if !REJECTION_VERDICTS
            .iter()
            .any(|verdict| upper.contains(verdict))
        {
            continue;
        }
        let haystack = format!("{}\n{}", row.header, row.body).to_lowercase();
        if !terms.iter().all(|term| haystack.contains(term)) {
            continue;
        }
        let decidable = EVIDENCE_MARKERS
            .iter()
            .any(|marker| haystack.contains(marker));
        let summary = format!(
            "docs/NEGATIVE_EVIDENCE.md:{} — {}",
            row.line,
            row.header.chars().take(140).collect::<String>()
        );
        if decidable {
            binding.push(summary);
        } else {
            void_prior.push(summary);
        }
    }

    if !binding.is_empty() {
        println!("BLOCKED — {} binding prior rejection(s):", binding.len());
        for row in &binding {
            println!("  {row}");
        }
        println!(
            "\nEach records why it was decidable (A/A null, counted mechanism, accuracy \
             refutation, or a large-magnitude loss). Do not re-derive these. If you believe one \
             is wrong, reopen it explicitly in the ledger with new evidence rather than silently \
             retrying it."
        );
        if !void_prior.is_empty() {
            println!("\nAlso {} VOID prior row(s) — not binding:", void_prior.len());
            for row in &void_prior {
                println!("  {row}");
            }
        }
        std::process::exit(2);
    }

    if !void_prior.is_empty() {
        println!(
            "VOID PRIOR — {} matching rejection(s), none of them binding:",
            void_prior.len()
        );
        for row in &void_prior {
            println!("  {row}");
        }
        println!(
            "\nThese rows record no A/A null, no counted mechanism, and no accuracy or \
             large-magnitude refutation, so they could not distinguish the lever from the \
             harness. They do NOT close this lever. Proceed — and record an A/A null control \
             this time, or tests/ledger_integrity.rs will reject your new row."
        );
        std::process::exit(0);
    }

    println!(
        "CLEAR — no prior REJECT row matches {:?}. Record an A/A null control when you write \
         your result.",
        terms
    );
}
