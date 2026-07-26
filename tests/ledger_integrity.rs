//! Ledger-integrity guard — makes a non-provable REJECT impossible to land.
//!
//! Fleet campaign `perf-campaign-20260725`, Meta-Lever #1, broadcast 2.
//!
//! A changed rejection must carry either a numerical same-invocation A/A null
//! or a counted unchanged-work mechanism. Accuracy prose, a large regression,
//! a profile, and CV alone are not write-gate exceptions. A changed KEEP must
//! carry a 64-hex benchmark-binary/ELF SHA-256. The pre-commit hook applies
//! these rules to the staged index, including backdated rows; this test pins the
//! parser and both sides of the contract.

use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "../examples/ledger_preflight.rs"]
mod ledger_preflight;

/// Rows dated on or after this must be provable. Chosen as the date the guard
/// landed, so it constrains the future without rewriting the past.
const ENFORCED_FROM: &str = "2026-07-26";

fn ledger_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/NEGATIVE_EVIDENCE.md")
}

/// One `## `-delimited ledger entry.
struct Entry {
    line: usize,
    date: String,
    header: String,
    body_lower: String,
}

fn parse_entries(text: &str) -> Vec<Entry> {
    let mut entries: Vec<Entry> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        if let Some(rest) = raw.strip_prefix("## ") {
            let date = rest
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_owned();
            entries.push(Entry {
                line: idx + 1,
                date,
                header: raw.to_owned(),
                body_lower: String::new(),
            });
        } else if let Some(current) = entries.last_mut() {
            current.body_lower.push_str(&raw.to_lowercase());
            current.body_lower.push('\n');
        }
    }
    entries
}

/// Verdict words this repo actually uses to close a lever.
///
/// Matching only `REJECT` would have missed **half the population** — 139 rows
/// instead of 277 — because rejections here live in prose titles, not a verdict
/// column: the `int4 mlp_0` family is closed under *DEAD*, *CLOSED*,
/// *FALSIFIED* and *NEGATIVE*, never under "REJECT". Anything narrower also
/// leaves the guard trivially bypassable by writing "DEAD" instead.
const REJECTION_VERDICTS: &[&str] = &[
    "REJECT",
    "DEAD",
    "CLOSED",
    "FALSIFIED",
    "NO-SHIP",
    "DO-NOT-RETRY",
    "NEGATIVE",
];

/// A rejection verdict, as opposed to a KEEP or a survey row.
fn is_reject(header: &str) -> bool {
    let upper = header.to_uppercase();
    REJECTION_VERDICTS
        .iter()
        .any(|verdict| upper.contains(verdict))
}

/// Does the row record anything that makes its rejection decidable?
fn has_evidence(entry: &Entry) -> bool {
    let text = format!("{}\n{}", entry.header, entry.body_lower);
    ledger_preflight::has_same_invocation_aa(&text)
        || ledger_preflight::has_counted_mechanism(&text)
}

/// Is this header dated `YYYY-MM-DD`? Section dividers and prose headers are not
/// ledger entries and must not be compared against the cutoff — a header like
/// `## previously: …` sorts *above* any date string and would be wrongly
/// enforced.
fn is_dated(date: &str) -> bool {
    let bytes = date.as_bytes();
    bytes.len() >= 10
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[7] == b'-'
        && bytes[8..10].iter().all(u8::is_ascii_digit)
}

#[test]
fn every_new_reject_row_records_why_it_is_decidable() {
    let text = std::fs::read_to_string(ledger_path()).expect("read docs/NEGATIVE_EVIDENCE.md");
    let entries = parse_entries(&text);

    let mut offenders = Vec::new();

    // Only dated `## YYYY-MM-DD …` headers are ledger rows. Prose headers and
    // section dividers are skipped outright — one of them reads
    // `## previously: blocked/neutral/rejected evidence`, which matches on
    // "rejected" and would otherwise be counted as a non-compliant row forever.
    for entry in entries
        .iter()
        .filter(|e| is_reject(&e.header) && is_dated(&e.date))
    {
        if entry.date.as_str() < ENFORCED_FROM || has_evidence(entry) {
            continue;
        }
        offenders.push(format!(
            "  docs/NEGATIVE_EVIDENCE.md:{} — {}",
            entry.line,
            entry.header.chars().take(120).collect::<String>()
        ));
    }

    assert!(
        offenders.is_empty(),
        "REJECT rows dated on/after {ENFORCED_FROM} record no evidence that the rejection was \
         decidable.\n\nA rejection needs either (1) a numerical same-invocation A/A null \
         control or (2) a counted unchanged-work mechanism. Accuracy prose, a large \
         regression, profile evidence, and CV alone do not satisfy the write gate. See \
         docs/LEDGER_RESURRECTION.md.\n\nOffending rows:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_ledger_is_parseable_and_non_trivial() {
    // Guards the guard: a parser that silently matches nothing would make the
    // test above vacuously green.
    let text = std::fs::read_to_string(ledger_path()).expect("read docs/NEGATIVE_EVIDENCE.md");
    let entries = parse_entries(&text);
    let rejects = entries.iter().filter(|e| is_reject(&e.header)).count();

    assert!(
        entries.len() > 500,
        "parsed only {} ledger entries — the '## ' entry format changed and this guard is no \
         longer reading the ledger",
        entries.len()
    );
    assert!(
        rejects > 200,
        "parsed only {rejects} REJECT rows out of {} entries — the verdict format changed and \
         this guard would pass vacuously",
        entries.len()
    );
}

#[test]
fn staged_reject_contract_is_strict_and_two_sided() {
    let invalid = "## 2026-07-26 - test: **REJECT — 0.40x and accuracy drift.**\n\
                   Profile self-time 20%; Amdahl ceiling 1.25x; CV 1%.\n";
    let violations =
        ledger_preflight::validate_changed_text("", invalid, "docs/NEGATIVE_EVIDENCE.md");
    assert_eq!(
        violations.len(),
        1,
        "magnitude, accuracy, profile, and CV must not bypass the write gate"
    );

    let negated = "## 2026-07-26 - test: **REJECT — flat.**\n\
                   No A/A null control was recorded; candidate median 1.001.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text("", negated, "docs/NEGATIVE_EVIDENCE.md").len(),
        1,
        "mentioning a missing null must not count as a null"
    );

    let valid_null = "## 2026-07-26 - test: **REJECT — flat.**\n\
                      Same-invocation A/A null control median 1.001, bootstrap CI95 \
                      [0.992, 1.009]. Candidate median 1.002.\n";
    assert!(
        ledger_preflight::validate_changed_text("", valid_null, "docs/NEGATIVE_EVIDENCE.md")
            .is_empty()
    );

    let valid_mechanism = "## 2026-07-26 - test: **REJECT — mechanism unchanged.**\n\
                           Instructions unchanged at 41024 in both arms; allocation count \
                           unchanged at 3.\n";
    assert!(
        ledger_preflight::validate_changed_text("", valid_mechanism, "docs/NEGATIVE_EVIDENCE.md")
            .is_empty()
    );
}

#[test]
fn staged_keep_requires_binary_or_elf_sha_not_an_output_oracle() {
    let oracle_only = "## 2026-07-26 - test: **KEEP — candidate wins.**\n\
                       Output oracle SHA-256 \
                       0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text("", oracle_only, "docs/NEGATIVE_EVIDENCE.md").len(),
        1
    );

    let binary = "## 2026-07-26 - test: **KEEP — candidate wins.**\n\
                  The executable ELF SHA-256 self-report is \
                  0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.\n";
    assert!(
        ledger_preflight::validate_changed_text("", binary, "docs/NEGATIVE_EVIDENCE.md").is_empty()
    );
}

#[test]
fn unchanged_legacy_rows_are_grandfathered_but_modified_rows_are_checked() {
    let legacy = "## 2026-06-01 - test: **REJECT — 1.00x.**\nNo null.\n";
    assert!(
        ledger_preflight::validate_changed_text(legacy, legacy, "docs/NEGATIVE_EVIDENCE.md")
            .is_empty()
    );
    let modified = "## 2026-06-01 - test: **REJECT — 1.01x after retry.**\nStill no null.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text(legacy, modified, "docs/NEGATIVE_EVIDENCE.md")
            .len(),
        1,
        "the staged comparison must prevent backdating"
    );
}

#[test]
fn nested_subsections_remain_inside_their_parent_row() {
    let rows = ledger_preflight::parse_rows(
        "## 2026-07-26 - test: **REJECT — sample.**\n### A/A evidence\nbody\n\
         ## 2026-07-26 - test: **KEEP — next.**\nbody\n",
    );
    assert_eq!(rows.len(), 2);
    assert!(rows[0].body.contains("### A/A evidence"));
}
