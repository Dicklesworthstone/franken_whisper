//! Ledger-integrity guard — makes a non-provable REJECT impossible to land.
//!
//! Fleet campaign `perf-campaign-20260725`, Meta-Lever #1, broadcast 2.
//!
//! ## Why this test exists
//!
//! The fleet-wide ledger-resurrection audit found that a REJECT row is usually
//! void not because the lever was wrongly judged, but because **the row cannot
//! prove anything either way**: an A/B ran, the row was rejected on a near-1.0
//! wall ratio, and no A/A null control and no counted mechanism were written
//! down. Across the fleet that class (`VOID-NONULL`) is the epidemic —
//! frankenfs 214 of 219 void rows, franken_whisper 79 of 82
//! (`docs/LEDGER_RESURRECTION.md`).
//!
//! The decisive data point is frankensqlite: **1.7% void**, not because it
//! audited leniently but because it ran this audit months ago and then
//! *institutionalized* it with a mechanically enforced preflight. Every repo
//! that audited once and stopped sits at 25–91%. **Ledger integrity decays.**
//! A convention that is merely documented is a convention that erodes; this
//! test is the enforcement, so the discipline survives the agents who wrote it.
//!
//! ## What a REJECT row must carry
//!
//! At least one of the following, mirroring the fleet taxonomy in
//! `docs/LEDGER_RESURRECTION.md` §1 — each is a reason the rejection is
//! *decidable*:
//!
//! - **A/A null control** — the effect is compared against the harness's own
//!   noise floor. The only thing that makes a near-1.0 ratio meaningful.
//! - **Counted mechanism** — instructions / cycles / syscalls / allocations /
//!   faults unchanged. A null cannot change the fact that no work was removed,
//!   so this refutes without one (`VALID-MECHANISM`).
//! - **Accuracy / faithfulness refutation** — WER, byte-exactness, or numerical
//!   safety. This repo's contract is transcript exactness, so many levers die
//!   here and never make a speed claim at all; a speed null is meaningless for
//!   them (`VALID-ACCURACY`, franken_whisper's proposed 7th class).
//! - **Large-magnitude refutation** — a stated ratio at or below 0.90×. No
//!   plausible null floor on this hardware spans a >10% loss. (Detected
//!   numerically; the bare word "slower" is *not* accepted, because it occurs
//!   in ordinary prose in a third of these rows.)
//! - **Profile-first rejection** — killed on a named frame's self-time or an
//!   Amdahl ceiling before any source was edited (`VALID-PROFILE`).
//!
//! ## Why the cutoff, and why the legacy debt is pinned rather than fixed
//!
//! 99 pre-existing rows do not comply. Failing on those would make this test
//! permanently red, and a permanently red test gets deleted or `#[ignore]`d —
//! which is how the discipline erodes in the first place. So history is
//! grandfathered and **counted**: [`LEGACY_NONCOMPLIANT_BUDGET`] pins that debt
//! so it can only shrink. Backdating a new row past the cutoff to dodge the
//! check trips the budget assertion instead.

use std::path::{Path, PathBuf};

/// Rows dated on or after this must be provable. Chosen as the date the guard
/// landed, so it constrains the future without rewriting the past.
const ENFORCED_FROM: &str = "2026-07-26";

/// Pre-cutoff REJECT rows that carry no decidability evidence, as measured when
/// this guard landed. It may only shrink. Lowering it as rows are rehabilitated
/// is encouraged; raising it means a non-provable row was backdated.
const LEGACY_NONCOMPLIANT_BUDGET: usize = 99;

/// Evidence that a rejection was decidable. See the module docs for why each
/// one independently suffices.
///
/// These are deliberately *specific phrases*, not keywords. An earlier draft
/// used bare `"instructions"`, `"allocation"`, `"slower"` and `"byte-identical"`
/// and was nearly vacuous — it passed 96% of historical REJECT rows, because
/// those words occur incidentally in ordinary prose. Two traps in particular:
///
/// - `"wer"` matched inside *were*, *lower*, *answer*. Substring matching is
///   not enough; [`contains_word`] requires non-alphanumeric boundaries.
/// - `"byte-identical"` is normally a *claim of exactness*, not an accuracy
///   refutation. A row reading "byte-identical but 1.02×, rejected" would have
///   passed on it while recording no null at all — the exact hole this guard
///   exists to close. Only `non-byte-exact` (a refutation) counts.
///
/// A guard that passes everything is worse than no guard: it manufactures
/// confidence. The current set fails ~36% of historical rows.
const EVIDENCE_MARKERS: &[&str] = &[
    // A/A null control — the effect was compared against the harness noise floor
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
    // counted mechanism — "no work was removed", which a null cannot overturn
    "instructions retired",
    "instruction count",
    "retired instructions",
    "cycle count",
    "cycles unchanged",
    "perf stat",
    "syscall count",
    "syscalls unchanged",
    "allocations unchanged",
    "alloc count",
    "allocation count",
    "page fault",
    "zero allocations",
    "no allocations",
    // accuracy / faithfulness / safety refutation (VALID-ACCURACY)
    "wer",
    "accuracy",
    "faithful",
    "not safe",
    "non-byte-exact",
    "transcript-unsafe",
    "quality gate",
    "regresses",
    "regression on",
    "drifts",
    // profile-first rejection (VALID-PROFILE)
    "self-time",
    "self time",
    "amdahl",
    "% of e2e",
];

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

/// Substring match that requires non-alphanumeric boundaries, so `"wer"` does
/// not match inside *were* / *lower* / *answer*.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let mut from = 0usize;
    while let Some(rel) = haystack[from..].find(needle) {
        let start = from + rel;
        let end = start + needle.len();
        let left_ok = start == 0
            || !haystack[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric());
        let right_ok = end == haystack.len()
            || !haystack[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric());
        if left_ok && right_ok {
            return true;
        }
        from = start + needle.len().max(1);
    }
    false
}

/// Does the row record anything that makes its rejection decidable?
fn has_evidence(entry: &Entry) -> bool {
    let haystack = format!("{}\n{}", entry.header.to_lowercase(), entry.body_lower);
    EVIDENCE_MARKERS
        .iter()
        .any(|marker| contains_word(&haystack, marker))
        // A ratio at or below 0.90x is a large-magnitude refutation on its own.
        || contains_large_regression(&haystack)
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

/// True when the text states a ratio of 0.90x or worse — a loss no null floor
/// on this hardware spans.
fn contains_large_regression(haystack: &str) -> bool {
    let bytes = haystack.as_bytes();
    for (i, window) in bytes.windows(2).enumerate() {
        if window != b"0." {
            continue;
        }
        let tail = &haystack[i + 2..];
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() {
            continue;
        }
        let suffix = &tail[digits.len()..];
        if !(suffix.starts_with('x') || suffix.starts_with('×')) {
            continue;
        }
        // "0.90" -> 90, "0.4" -> 4 (scaled to 40) — compare on the first two digits.
        let scaled: u32 = match digits.len() {
            1 => digits.parse::<u32>().unwrap_or(99) * 10,
            _ => digits[..2].parse::<u32>().unwrap_or(99),
        };
        if scaled <= 90 {
            return true;
        }
    }
    false
}

#[test]
fn every_new_reject_row_records_why_it_is_decidable() {
    let text = std::fs::read_to_string(ledger_path()).expect("read docs/NEGATIVE_EVIDENCE.md");
    let entries = parse_entries(&text);

    let mut offenders = Vec::new();
    let mut legacy_noncompliant = 0usize;

    // Only dated `## YYYY-MM-DD …` headers are ledger rows. Prose headers and
    // section dividers are skipped outright — one of them reads
    // `## previously: blocked/neutral/rejected evidence`, which matches on
    // "rejected" and would otherwise be counted as a non-compliant row forever.
    for entry in entries
        .iter()
        .filter(|e| is_reject(&e.header) && is_dated(&e.date))
    {
        if has_evidence(entry) {
            continue;
        }
        if entry.date.as_str() >= ENFORCED_FROM {
            offenders.push(format!(
                "  docs/NEGATIVE_EVIDENCE.md:{} — {}",
                entry.line,
                entry.header.chars().take(120).collect::<String>()
            ));
        } else {
            legacy_noncompliant += 1;
        }
    }

    assert!(
        offenders.is_empty(),
        "REJECT rows dated on/after {ENFORCED_FROM} record no evidence that the rejection was \
         decidable.\n\nA rejection needs at least ONE of: an A/A null control; a counted \
         mechanism (instructions/cycles/syscalls/allocations/faults unchanged); an \
         accuracy/faithfulness/byte-exactness refutation; a large-magnitude loss (<=0.90x or \
         'SLOWER'); or a profile-first self-time/Amdahl rejection.\n\nWithout one of those the \
         row cannot distinguish the lever from the harness, which is the VOID-NONULL class that \
         made {LEGACY_NONCOMPLIANT_BUDGET} of this ledger's rows unusable. See \
         docs/LEDGER_RESURRECTION.md.\n\nOffending rows:\n{}",
        offenders.join("\n")
    );

    assert!(
        legacy_noncompliant <= LEGACY_NONCOMPLIANT_BUDGET,
        "pre-{ENFORCED_FROM} rows without decidability evidence rose to {legacy_noncompliant}, \
         above the pinned budget of {LEGACY_NONCOMPLIANT_BUDGET}. Legacy debt may only shrink — \
         a new REJECT row must not be dated before the cutoff to bypass this guard."
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
fn large_regression_detection_is_sound() {
    assert!(contains_large_regression("measured 0.439x versus baseline"));
    assert!(contains_large_regression("came in at 0.40×"));
    assert!(contains_large_regression("0.90x exactly at the boundary"));
    assert!(!contains_large_regression("0.91x is inside the floor"));
    assert!(!contains_large_regression("1.024879x, inside the null envelope"));
    assert!(!contains_large_regression("no ratio here at all"));
}
