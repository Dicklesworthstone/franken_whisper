//! Ledger-integrity guard — makes a non-provable REJECT impossible to land.
//!
//! Fleet campaign `perf-campaign-20260725`, Meta-Lever #1, broadcast 2.
//!
//! A changed rejection must carry either a numerical same-invocation A/A null
//! or a counted unchanged-work mechanism. Accuracy prose, a large regression,
//! a profile, and CV alone are not write-gate exceptions. A changed KEEP/WIN
//! must carry a 64-hex benchmark-binary/ELF SHA-256 and classify a measured
//! speed result as either a maintenance self-speedup or a same-invocation
//! actual-incumbent campaign win. The pre-commit hook applies these rules to
//! the staged index, including backdated rows; this test pins the parser and
//! both sides of the contract.

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
/// Matching only `REJECT` would have missed **half the mechanical screen** —
/// 139 candidate headers instead of 277 — because closure words here live in
/// prose titles, not a verdict column: the `int4 mlp_0` family is closed under
/// *DEAD*, *CLOSED*, *FALSIFIED* and *NEGATIVE*, never under "REJECT".
/// Hand-adjudication later reduced those 277 candidates to 188 actual
/// performance rejections. Anything narrower also leaves the guard trivially
/// bypassable by writing "DEAD" instead.
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
    let reject_candidates = entries.iter().filter(|e| is_reject(&e.header)).count();

    assert!(
        entries.len() > 500,
        "parsed only {} ledger entries — the '## ' entry format changed and this guard is no \
         longer reading the ledger",
        entries.len()
    );
    assert!(
        reject_candidates > 200,
        "parsed only {reject_candidates} candidate verdict headers out of {} entries — the \
         verdict format changed and this guard would pass vacuously",
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

    let unrelated_unchanged = "## 2026-07-26 - test: **REJECT — mechanism unclear.**\n\
                               Allocations increased from 3 to 5, while transcript output \
                               was unchanged.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text(
            "",
            unrelated_unchanged,
            "docs/NEGATIVE_EVIDENCE.md"
        )
        .len(),
        1,
        "an unchanged output must not launder a changed counter"
    );

    let unquantified_null = "## 2026-07-26 - test: **REJECT — flat.**\n\
                             Same-binary A/A null control was mentioned; candidate median \
                             1.001, but no numerical null statistic was recorded.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text("", unquantified_null, "docs/NEGATIVE_EVIDENCE.md")
            .len(),
        1,
        "a candidate statistic must not masquerade as a numerical null"
    );

    let candidate_laundering = "## 2026-07-26 - test: **REJECT — flat.**\n\
                                Same-invocation A/A null control ran; candidate median 1.001.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text(
            "",
            candidate_laundering,
            "docs/NEGATIVE_EVIDENCE.md"
        )
        .len(),
        1,
        "a candidate ratio in another clause must not count as the null statistic"
    );

    let duration_laundering = "## 2026-07-26 - test: **REJECT — flat.**\n\
                               Same-invocation A/A null control median 10.0 ms.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text(
            "",
            duration_laundering,
            "docs/NEGATIVE_EVIDENCE.md"
        )
        .len(),
        1,
        "a multi-digit duration must not be mistaken for a null ratio"
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

    let missing_binary_with_oracle = "## 2026-07-26 - test: **KEEP — candidate wins.**\n\
                                      Binary SHA unavailable. Output oracle SHA-256 \
                                      0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.\n";
    assert_eq!(
        ledger_preflight::validate_changed_text(
            "",
            missing_binary_with_oracle,
            "docs/NEGATIVE_EVIDENCE.md"
        )
        .len(),
        1,
        "a nearby output digest must not launder an explicitly missing binary digest"
    );
}

#[test]
fn staged_speed_keeps_distinguish_maintenance_from_incumbent_wins() {
    let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let incumbent_digest = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    let unclassified = format!(
        "## 2026-07-27 - test: **KEEP — 1.20x faster.**\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert_eq!(
        ledger_preflight::validate_changed_text("", &unclassified, "docs/PERF_LEDGER.md").len(),
        1,
        "a speed KEEP without a result class must be blocked"
    );

    let maintenance = format!(
        "## 2026-07-27 - test: **KEEP — 1.20x faster.**\n\
         Result class: SELF-SPEEDUP / MAINTENANCE.\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert!(
        ledger_preflight::validate_changed_text("", &maintenance, "docs/PERF_LEDGER.md").is_empty(),
        "a labeled self-speedup may justify a maintenance KEEP"
    );

    let self_as_campaign = format!(
        "## 2026-07-27 - test: **CAMPAIGN WIN — 1.20x faster.**\n\
         Result class: SELF-SPEEDUP / MAINTENANCE.\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert_eq!(
        ledger_preflight::validate_changed_text("", &self_as_campaign, "docs/PERF_LEDGER.md").len(),
        1,
        "a self-speedup cannot be presented as campaign output"
    );

    let same_session_only = format!(
        "## 2026-07-27 - test: **KEEP — incumbent comparison 1.20x.**\n\
         Result class: INCUMBENT-WIN / CAMPAIGN WIN.\n\
         Legacy incumbent: whisper.cpp whisper-cli.\n\
         Incumbent binary SHA-256: {incumbent_digest}.\n\
         Comparator execution: same session but not interleaved.\n\
         Measured incumbent ratio: 1.20x.\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert_eq!(
        ledger_preflight::validate_changed_text("", &same_session_only, "docs/PERF_LEDGER.md")
            .len(),
        1,
        "same-session separate runs are not a campaign win"
    );

    let proxy_incumbent = format!(
        "## 2026-07-27 - test: **KEEP — incumbent comparison 1.20x.**\n\
         Result class: INCUMBENT-WIN / CAMPAIGN WIN.\n\
         Legacy incumbent: proxy implementation.\n\
         Incumbent binary SHA-256: {incumbent_digest}.\n\
         Comparator execution: actual incumbent side-by-side in the same invocation.\n\
         Measured incumbent ratio: 1.20x.\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert_eq!(
        ledger_preflight::validate_changed_text("", &proxy_incumbent, "docs/PERF_LEDGER.md").len(),
        1,
        "a named proxy is not the actual legacy incumbent"
    );

    let missing_incumbent_identity = format!(
        "## 2026-07-27 - test: **KEEP — incumbent comparison 1.20x.**\n\
         Result class: INCUMBENT-WIN / CAMPAIGN WIN.\n\
         Legacy incumbent: whisper.cpp whisper-cli.\n\
         Comparator execution: actual legacy incumbent side-by-side in the same invocation.\n\
         Measured incumbent ratio: 1.20x.\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert_eq!(
        ledger_preflight::validate_changed_text(
            "",
            &missing_incumbent_identity,
            "docs/PERF_LEDGER.md"
        )
        .len(),
        1,
        "a campaign win must identify the actual incumbent binary"
    );

    let missing_candidate_identity = format!(
        "## 2026-07-27 - test: **KEEP — incumbent comparison 1.20x.**\n\
         Result class: INCUMBENT-WIN / CAMPAIGN WIN.\n\
         Legacy incumbent: whisper.cpp whisper-cli.\n\
         Incumbent binary SHA-256: {incumbent_digest}.\n\
         Comparator execution: actual legacy incumbent side-by-side in the same invocation.\n\
         Measured incumbent ratio: 1.20x.\n"
    );
    assert_eq!(
        ledger_preflight::validate_changed_text(
            "",
            &missing_candidate_identity,
            "docs/PERF_LEDGER.md"
        )
        .len(),
        1,
        "the incumbent digest must not masquerade as the candidate harness digest"
    );

    let live_incumbent = format!(
        "## 2026-07-27 - test: **KEEP — incumbent comparison 1.20x.**\n\
         Result class: INCUMBENT-WIN / CAMPAIGN WIN.\n\
         Legacy incumbent: whisper.cpp whisper-cli.\n\
         Incumbent binary SHA-256: {incumbent_digest}.\n\
         Comparator execution: actual legacy incumbent side-by-side in the same invocation.\n\
         Measured incumbent ratio: 1.20x.\n\
         Executable ELF SHA-256 {digest}.\n"
    );
    assert!(
        ledger_preflight::validate_changed_text("", &live_incumbent, "docs/PERF_LEDGER.md")
            .is_empty(),
        "a named live incumbent arm in the same invocation satisfies the campaign contract"
    );

    let informational = "## 2026-07-27 - test: NON-CAMPAIGN COMPARISON — 1.20x.\n\
                         Result class: NON-CAMPAIGN / INFORMATIONAL.\n";
    assert!(
        ledger_preflight::validate_changed_text("", informational, "docs/PERF_LEDGER.md")
            .is_empty(),
        "an explicitly non-campaign point estimate may remain internal without a positive verdict"
    );
}

#[test]
fn public_docs_reject_performance_retraction_narratives() {
    let withdrawn = "Current result: 1.10x.\nWithdrawn claim: an older number was wrong.\n";
    assert_eq!(
        ledger_preflight::validate_public_changed_text("", withdrawn, "README.md").len(),
        1
    );

    let current_only =
        "Current live-incumbent same-invocation result: 1.10x faster than whisper.cpp.\n";
    assert!(
        ledger_preflight::validate_public_changed_text("", current_only, "README.md").is_empty()
    );

    let domain_event = "`transcript.retract` replaces a speculative partial transcript.\n";
    assert!(
        ledger_preflight::validate_public_changed_text("", domain_event, "README.md").is_empty(),
        "the public-doc policy must not confuse the product's retract event with claim retractions"
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

#[test]
fn explicit_retry_heading_beats_an_earlier_historical_mention() {
    let mut rows = ledger_preflight::parse_rows(
        "## 2026-07-26 - test: **NO VERDICT.**\n\
         The old retry predicate required a warm worker, and that condition was satisfied.\n\n\
         **Concrete retry predicate:** rerun only after the faithful baseline exists.\n\
         Gate on the same-invocation A/A median CI.\n",
    );
    let row = rows.pop().expect("one ledger row");
    let predicate = ledger_preflight::retry_predicate(&row);

    assert!(predicate.contains("faithful baseline"));
    assert!(predicate.contains("same-invocation A/A"));
    assert!(!predicate.contains("warm worker"));
}

/// New positive ledger rows must pass the same parser as the pre-commit hook.
///
/// This is deliberately an actual-ledger self-check, not another regex taxonomy:
/// the first version searched only `NEGATIVE_EVIDENCE.md`, while the campaign win
/// lived in `PERF_LEDGER.md`, so it could pass without inspecting the row it was
/// intended to protect.
const CLASS_ENFORCED_FROM: &str = "2026-07-27";

#[test]
fn new_keep_rows_declare_self_speedup_or_vs_incumbent() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0usize;
    let mut violations = Vec::new();

    for (relative, path) in [
        (
            "docs/NEGATIVE_EVIDENCE.md",
            root.join("docs/NEGATIVE_EVIDENCE.md"),
        ),
        ("docs/PERF_LEDGER.md", root.join("docs/PERF_LEDGER.md")),
    ] {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                violations.push(format!("{relative}: unable to read ledger: {error}"));
                continue;
            }
        };
        for row in ledger_preflight::parse_rows(&text) {
            let date = row
                .header
                .strip_prefix("## ")
                .and_then(|rest| rest.split_whitespace().next())
                .unwrap_or_default();
            if !is_dated(date) || date < CLASS_ENFORCED_FROM || is_reject(&row.header) {
                continue;
            }
            let upper = row.header.to_uppercase();
            let positive = upper.contains("KEEP")
                || upper.contains(" WIN ")
                || upper.contains("— WIN")
                || upper.contains("LAND ")
                || upper.contains("LANDED")
                || upper.contains("SHIPPED");
            if !positive {
                continue;
            }
            checked += 1;
            let row_text = format!("{}\n{}", row.header, row.body);
            violations.extend(ledger_preflight::validate_changed_text(
                "", &row_text, relative,
            ));
        }
    }

    assert!(
        checked > 0,
        "the result-class ledger self-check was vacuous"
    );
    assert!(
        violations.is_empty(),
        "positive ledger rows dated on/after {CLASS_ENFORCED_FROM} fail the staged preflight \
         contract:\n{}",
        violations.join("\n")
    );
}
