//! Ledger preflight and staged-result gate.
//!
//! Fleet campaign `perf-campaign-20260725`, Meta-Lever #1, broadcast 2.
//! Modelled on frankensqlite's `sql_pipeline_candidate_preflight`.
//!
//! `surface` searches the negative-evidence ledger before source is touched,
//! prints matching retry predicates, and exits 2 for a binding prior result.
//! `validate-staged` compares the Git index with HEAD and exits 2 if a changed
//! rejection has neither a same-invocation A/A null nor a counted mechanism,
//! if a changed KEEP/WIN lacks a benchmark-binary/ELF SHA-256 or result class,
//! if a campaign win lacks a live same-invocation incumbent arm, or if a
//! changed public performance document adds retraction narrative.
//!
//! Exit 0 means clear. Exit 2 means BLOCKED. Other non-zero exits are usage or
//! infrastructure failures.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const REJECTION_VERDICTS: &[&str] = &[
    "REJECT",
    "DEAD",
    "CLOSED",
    "FALSIFIED",
    "NO-SHIP",
    "DO-NOT-RETRY",
    "NEGATIVE",
];
const POSITIVE_VERDICTS: &[&str] = &["KEEP", "WIN", "LAND", "LANDED", "SHIPPED"];

const NULL_MARKERS: &[&str] = &["a/a", "null control", "identity null", "base/base"];
const SAME_INVOCATION_MARKERS: &[&str] = &[
    "same invocation",
    "same-invocation",
    "same binary",
    "same-binary",
    "same elf",
    "same-elf",
];
const NULL_STATISTIC_MARKERS: &[&str] = &[
    "median",
    "ci95",
    "ci 95",
    "confidence interval",
    "bootstrap",
];
const NEGATED_NULL_MARKERS: &[&str] = &[
    "no a/a",
    "without a/a",
    "no null control",
    "without a null control",
    "null control unavailable",
    "null control not recorded",
    "missing null control",
    "no numerical null",
    "numerical null statistic not recorded",
    "null statistic not recorded",
    "missing null statistic",
];
const COUNTED_NOUNS: &[&str] = &[
    "instructions",
    "instruction count",
    "cycles",
    "cycle count",
    "syscalls",
    "syscall count",
    "allocations",
    "allocation count",
    "page faults",
    "fault count",
    "bytes moved",
    "bytes read",
    "bytes written",
];
const UNCHANGED_MARKERS: &[&str] = &[
    "unchanged",
    "same count",
    "identical count",
    "zero delta",
    "did not change",
    "no change",
    "equal count",
];
const BINARY_SHA_MARKERS: &[&str] = &[
    "benchmark binary sha",
    "benchmark-binary sha",
    "binary sha",
    "elf sha",
    "executable sha",
    "probe_elf_sha256",
];
const NEGATED_BINARY_SHA_MARKERS: &[&str] = &[
    "no benchmark binary sha",
    "without a benchmark binary sha",
    "benchmark binary sha missing",
    "benchmark binary sha unavailable",
    "benchmark binary sha not recorded",
    "no binary sha",
    "without a binary sha",
    "binary sha missing",
    "binary sha unavailable",
    "binary sha not recorded",
    "no elf sha",
    "without an elf sha",
    "elf sha missing",
    "elf sha unavailable",
    "elf sha not recorded",
];
const CANDIDATE_BINARY_SHA_MARKERS: &[&str] = &[
    "benchmark binary sha",
    "benchmark-binary sha",
    "candidate binary sha",
    "candidate-binary sha",
    "candidate elf sha",
    "harness elf sha",
    "harness_elf_sha256",
    "probe_elf_sha256",
    "executable elf sha",
    "executable sha",
];
const INCUMBENT_BINARY_SHA_MARKERS: &[&str] = &[
    "incumbent binary sha",
    "incumbent-binary sha",
    "incumbent_bin_sha256",
    "incumbent elf sha",
];
const NEGATED_INCUMBENT_BINARY_SHA_MARKERS: &[&str] = &[
    "incumbent binary sha missing",
    "incumbent binary sha unavailable",
    "incumbent binary sha not recorded",
    "no incumbent binary sha",
    "incumbent elf sha missing",
    "incumbent elf sha unavailable",
    "incumbent elf sha not recorded",
];
const LEDGER_PATHS: &[&str] = &["docs/NEGATIVE_EVIDENCE.md", "docs/PERF_LEDGER.md"];
const PUBLIC_DOC_PATHS: &[&str] = &[
    "README.md",
    "CHANGELOG.md",
    "docs/PERF_FRONTIER.md",
    "docs/native_engine_contract.md",
    "docs/cc_lane_finalization.md",
];
const PUBLIC_RETRACTION_MARKERS: &[&str] = &[
    "withdrawn claim",
    "withdrawn and replaced",
    "earlier revisions",
    "previously claimed",
    "previously we claimed",
    "we previously claimed",
    "old headline",
    "stale claim",
    "known gate defect",
    "published misinformation",
    "published 0.78",
    "was 0.78",
    "wrong figure",
    "wrong number",
];
const INCUMBENT_SAME_INVOCATION_MARKERS: &[&str] = &["same invocation", "same-invocation"];
const SIDE_BY_SIDE_MARKERS: &[&str] = &["side-by-side", "side by side", "interleaved"];
const NEGATED_INCUMBENT_EXECUTION_MARKERS: &[&str] = &[
    "not same invocation",
    "not in the same invocation",
    "separate invocation",
    "separate-invocation",
    "same session but",
    "not side-by-side",
    "not side by side",
    "not interleaved",
];

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) line: usize,
    pub(crate) header: String,
    pub(crate) body: String,
}

impl Row {
    fn text(&self) -> String {
        format!("{}\n{}", self.header, self.body)
    }
}

pub(crate) fn parse_rows(text: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        if raw.starts_with("## ") {
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
    rows
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn find_token(haystack: &str, needle: &str) -> Option<usize> {
    haystack.match_indices(needle).find_map(|(start, _)| {
        let bytes = haystack.as_bytes();
        let end = start + needle.len();
        let left_ok =
            start == 0 || (!bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_');
        let right_ok =
            end == bytes.len() || (!bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_');
        (left_ok && right_ok).then_some(start)
    })
}

fn contains_ratio_literal(text: &str) -> bool {
    let mut previous = None;
    for character in text.chars() {
        if matches!(character, 'x' | 'X' | '×')
            && previous.is_some_and(|prior: char| prior.is_ascii_digit())
        {
            return true;
        }
        previous = Some(character);
    }
    false
}

fn has_performance_claim(text: &str) -> bool {
    let lower = text.to_lowercase();
    contains_ratio_literal(&lower)
        || lower.contains("speedup")
        || lower.contains("faster")
        || lower.contains("latency reduction")
        || lower.contains("throughput gain")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResultClass {
    Missing,
    SelfSpeedup,
    IncumbentWin,
    NonCampaign,
    Ambiguous,
}

fn result_class(text: &str) -> ResultClass {
    let mut classes = Vec::new();
    for line in text.lines() {
        let lower = line.to_lowercase();
        let Some((_, value)) = lower.split_once("result class:") else {
            continue;
        };
        let self_speedup = value.contains("self-speedup") && value.contains("maintenance");
        let incumbent_win = (value.contains("incumbent-win") || value.contains("incumbent win"))
            && (value.contains("campaign-win") || value.contains("campaign win"));
        let non_campaign = value.contains("non-campaign") && value.contains("informational");
        match (self_speedup, incumbent_win, non_campaign) {
            (true, false, false) => classes.push(ResultClass::SelfSpeedup),
            (false, true, false) => classes.push(ResultClass::IncumbentWin),
            (false, false, true) => classes.push(ResultClass::NonCampaign),
            _ => classes.push(ResultClass::Ambiguous),
        }
    }
    match classes.as_slice() {
        [] => ResultClass::Missing,
        [class] => *class,
        _ => ResultClass::Ambiguous,
    }
}

fn has_named_legacy_incumbent(text: &str) -> bool {
    text.lines().any(|line| {
        let lower = line.to_lowercase();
        let Some((_, value)) = lower.split_once("legacy incumbent:") else {
            return false;
        };
        let value = value.trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '*' | '_' | '`')
        });
        !value.is_empty()
            && value.chars().any(char::is_alphanumeric)
            && !contains_any(
                value,
                &[
                    "none",
                    "n/a",
                    "unknown",
                    "tbd",
                    "proxy",
                    "self",
                    "candidate",
                ],
            )
    })
}

fn has_same_invocation_incumbent_arm(text: &str) -> bool {
    let lower = text.to_lowercase();
    let lines: Vec<&str> = lower.lines().collect();
    for start in 0..lines.len() {
        for end in (start + 1)..=(start + 3).min(lines.len()) {
            let window = lines[start..end].join(" ");
            if contains_any(&window, NEGATED_INCUMBENT_EXECUTION_MARKERS) {
                continue;
            }
            if window.contains("comparator execution:")
                && window.contains("incumbent")
                && contains_any(&window, INCUMBENT_SAME_INVOCATION_MARKERS)
                && contains_any(&window, SIDE_BY_SIDE_MARKERS)
            {
                return true;
            }
        }
    }
    false
}

fn has_measured_incumbent_ratio(text: &str) -> bool {
    text.lines().any(|line| {
        let lower = line.to_lowercase();
        lower.contains("measured incumbent ratio:") && contains_ratio_literal(&lower)
    })
}

pub(crate) fn has_same_invocation_aa(text: &str) -> bool {
    let lower = text.to_lowercase();
    let lines: Vec<&str> = lower.lines().collect();

    // Keep the same-invocation marker near the evidence, then require the null
    // label, statistic label, and ratio in one clause. Without the clause rule,
    // "A/A ran; candidate median 1.001" launders the candidate statistic into
    // a numerical null. Three adjacent lines accommodate Markdown wrapping.
    for start in 0..lines.len() {
        for end in (start + 1)..=(start + 3).min(lines.len()) {
            let window = lines[start..end].join("\n");
            if contains_any(&window, NEGATED_NULL_MARKERS) {
                continue;
            }
            if !contains_any(&window, SAME_INVOCATION_MARKERS) {
                continue;
            }
            if window.split(['\n', ';']).any(|clause| {
                contains_any(clause, NULL_MARKERS)
                    && contains_any(clause, NULL_STATISTIC_MARKERS)
                    && contains_null_ratio_literal(clause)
            }) {
                return true;
            }
        }
    }
    false
}

pub(crate) fn has_counted_mechanism(text: &str) -> bool {
    let lower = text.to_lowercase();
    // Require the counter, equality statement, and a concrete number in the
    // same short clause. Without this locality, prose such as "allocations
    // increased; transcript unchanged" incorrectly passes by combining two
    // unrelated claims.
    lower.split(['\n', ';', '|', ',', '.']).any(|clause| {
        contains_any(clause, COUNTED_NOUNS)
            && contains_any(clause, UNCHANGED_MARKERS)
            && clause.bytes().any(|byte| byte.is_ascii_digit())
    })
}

fn contains_null_ratio_literal(text: &str) -> bool {
    let bytes = text.as_bytes();
    (0..bytes.len().saturating_sub(2)).any(|start| {
        let left_is_boundary = start == 0
            || (!bytes[start - 1].is_ascii_digit()
                && bytes[start - 1] != b'.'
                && bytes[start - 1] != b'-');
        left_is_boundary
            && matches!(bytes[start], b'0' | b'1')
            && bytes[start + 1] == b'.'
            && bytes[start + 2].is_ascii_digit()
    })
}

fn contains_real_sha256(bytes: &[u8]) -> bool {
    if bytes.len() < 64 {
        return false;
    }
    for start in 0..=bytes.len() - 64 {
        let candidate = &bytes[start..start + 64];
        if !candidate.iter().all(u8::is_ascii_hexdigit) {
            continue;
        }
        let left_ok = start == 0 || !bytes[start - 1].is_ascii_hexdigit();
        let right_ok = start + 64 == bytes.len() || !bytes[start + 64].is_ascii_hexdigit();
        let non_placeholder = candidate.iter().any(|byte| *byte != candidate[0]);
        if left_ok && right_ok && non_placeholder {
            return true;
        }
    }
    false
}

pub(crate) fn has_binary_sha256(text: &str) -> bool {
    let lower = text.to_lowercase();
    let lines: Vec<&str> = lower.lines().collect();

    // The digest must live on the marker line or its wrapped continuation.
    // A broad character window allowed "binary SHA unavailable" followed by an
    // output-oracle digest later in the paragraph to launder a KEEP.
    for start in 0..lines.len() {
        for end in (start + 1)..=(start + 2).min(lines.len()) {
            let window = lines[start..end].join(" ");
            if contains_any(&window, NEGATED_BINARY_SHA_MARKERS) {
                continue;
            }
            // A campaign row carries two executable identities. The generic
            // "binary sha" marker must not let the incumbent digest masquerade
            // as the candidate/harness self-report.
            if contains_any(&window, INCUMBENT_BINARY_SHA_MARKERS)
                && !contains_any(&window, CANDIDATE_BINARY_SHA_MARKERS)
            {
                continue;
            }
            if contains_any(&window, BINARY_SHA_MARKERS) && contains_real_sha256(window.as_bytes())
            {
                return true;
            }
        }
    }
    false
}

fn has_incumbent_binary_sha256(text: &str) -> bool {
    let lower = text.to_lowercase();
    let lines: Vec<&str> = lower.lines().collect();
    for start in 0..lines.len() {
        for end in (start + 1)..=(start + 2).min(lines.len()) {
            let window = lines[start..end].join(" ");
            if contains_any(&window, NEGATED_INCUMBENT_BINARY_SHA_MARKERS) {
                continue;
            }
            if contains_any(&window, INCUMBENT_BINARY_SHA_MARKERS)
                && contains_real_sha256(window.as_bytes())
            {
                return true;
            }
        }
    }
    false
}

fn has_profile_evidence(text: &str) -> bool {
    let lower = text.to_lowercase();
    (lower.contains("self-time") || lower.contains("self time"))
        && (lower.contains("amdahl") || lower.contains("ceiling"))
        && lower.contains('%')
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Verdict {
    Keep,
    Reject,
    Other,
}

fn verdict(header: &str) -> Verdict {
    let upper = header.to_uppercase();
    let keep_at = POSITIVE_VERDICTS
        .iter()
        .filter_map(|word| find_token(&upper, word))
        .min();
    let reject_at = REJECTION_VERDICTS
        .iter()
        .filter_map(|word| find_token(&upper, word))
        .min();
    match (keep_at, reject_at) {
        (Some(keep), Some(reject)) if keep < reject => Verdict::Keep,
        (_, Some(_)) => Verdict::Reject,
        (Some(_), None) => Verdict::Keep,
        (None, None) => Verdict::Other,
    }
}

fn row_violation(row: &Row, path: &str) -> Option<String> {
    let text = row.text();
    let class = result_class(&text);
    let row_verdict = verdict(&row.header);
    if row_verdict == Verdict::Reject
        && !has_same_invocation_aa(&text)
        && !has_counted_mechanism(&text)
    {
        return Some(format!(
            "{path}:{} — changed rejection lacks BOTH a numerical same-invocation A/A \
             null and a counted unchanged-work mechanism: {}",
            row.line, row.header
        ));
    }

    let positive_result = row_verdict == Verdict::Keep
        || matches!(class, ResultClass::SelfSpeedup | ResultClass::IncumbentWin);
    if !positive_result {
        return None;
    }
    if !has_binary_sha256(&text) {
        return Some(format!(
            "{path}:{} — changed KEEP/WIN lacks a 64-hex benchmark-binary/ELF SHA-256: {}",
            row.line, row.header
        ));
    }
    if has_performance_claim(&text) && class == ResultClass::Missing {
        return Some(format!(
            "{path}:{} — changed performance KEEP/WIN lacks `Result class: \
             SELF-SPEEDUP / MAINTENANCE` or `Result class: INCUMBENT-WIN / CAMPAIGN WIN`: {}",
            row.line, row.header
        ));
    }
    if class == ResultClass::Ambiguous {
        return Some(format!(
            "{path}:{} — changed KEEP/WIN has an ambiguous or repeated `Result class:` field: {}",
            row.line, row.header
        ));
    }
    if class == ResultClass::NonCampaign {
        return Some(format!(
            "{path}:{} — NON-CAMPAIGN / INFORMATIONAL evidence cannot use a KEEP/WIN verdict: {}",
            row.line, row.header
        ));
    }
    let lower_header = row.header.to_lowercase();
    let lower_text = text.to_lowercase();
    if class == ResultClass::SelfSpeedup
        && (lower_header.contains("campaign win")
            || lower_header.contains("incumbent win")
            || contains_any(
                &lower_text,
                &[
                    "campaign output: yes",
                    "competitive result: yes",
                    "competitive claim: yes",
                ],
            ))
    {
        return Some(format!(
            "{path}:{} — a SELF-SPEEDUP / MAINTENANCE row cannot claim campaign or \
             competitive-win status: {}",
            row.line, row.header
        ));
    }
    if class == ResultClass::IncumbentWin && !has_named_legacy_incumbent(&text) {
        return Some(format!(
            "{path}:{} — campaign win lacks a concrete `Legacy incumbent:` name: {}",
            row.line, row.header
        ));
    }
    if class == ResultClass::IncumbentWin && !has_incumbent_binary_sha256(&text) {
        return Some(format!(
            "{path}:{} — campaign win lacks a 64-hex incumbent-binary/ELF SHA-256: {}",
            row.line, row.header
        ));
    }
    if class == ResultClass::IncumbentWin && !has_same_invocation_incumbent_arm(&text) {
        return Some(format!(
            "{path}:{} — campaign win lacks `Comparator execution:` proving the actual \
             incumbent ran side-by-side in the same invocation: {}",
            row.line, row.header
        ));
    }
    if class == ResultClass::IncumbentWin && !has_measured_incumbent_ratio(&text) {
        return Some(format!(
            "{path}:{} — campaign win lacks a numerical `Measured incumbent ratio:`: {}",
            row.line, row.header
        ));
    }
    None
}

pub(crate) fn validate_changed_text(head: &str, staged: &str, path: &str) -> Vec<String> {
    let old_rows: HashSet<String> = parse_rows(head).into_iter().map(|row| row.text()).collect();
    parse_rows(staged)
        .into_iter()
        .filter(|row| !old_rows.contains(&row.text()))
        .filter_map(|row| row_violation(&row, path))
        .collect()
}

pub(crate) fn validate_public_changed_text(head: &str, staged: &str, path: &str) -> Vec<String> {
    let old_lines: HashSet<&str> = head.lines().collect();
    staged
        .lines()
        .enumerate()
        .filter(|(_, line)| !old_lines.contains(line))
        .filter_map(|(index, line)| {
            let lower = line.to_lowercase();
            PUBLIC_RETRACTION_MARKERS
                .iter()
                .find(|marker| lower.contains(**marker))
                .map(|marker| {
                    format!(
                        "{path}:{} — public performance docs state the current admitted claim; \
                         retraction narrative `{marker}` belongs only in the internal ledgers",
                        index + 1
                    )
                })
        })
        .collect()
}

fn retry_marker_rank(line: &str) -> Option<u8> {
    let markers = [
        "concrete retry predicate",
        "retry predicate",
        "retry condition",
        "retry-condition",
        "retry only",
        "do not retry",
        "reopen only",
    ];
    let lower = line.to_lowercase();
    let normalized = lower
        .trim_start_matches(|character: char| {
            matches!(character, ' ' | '\t' | '*' | '_' | '#' | '-')
        })
        .trim_start();
    if markers.iter().any(|marker| normalized.starts_with(marker)) {
        return Some(0);
    }
    markers
        .iter()
        .any(|marker| lower.contains(marker))
        .then_some(1)
}

pub(crate) fn retry_predicate(row: &Row) -> String {
    let lines: Vec<&str> = row.body.lines().collect();
    for wanted_rank in [0, 1] {
        for (index, line) in lines.iter().enumerate() {
            if retry_marker_rank(line) != Some(wanted_rank) {
                continue;
            }
            let mut paragraph = Vec::new();
            for candidate in &lines[index..] {
                if candidate.trim().is_empty() && !paragraph.is_empty() {
                    break;
                }
                if candidate.starts_with('#') || candidate.trim() == "---" {
                    break;
                }
                paragraph.push(candidate.trim());
            }
            let joined = paragraph.join(" ");
            return joined.chars().take(900).collect();
        }
    }
    "(none recorded)".to_owned()
}

fn root() -> PathBuf {
    Path::new(option_env!("CARGO_MANIFEST_DIR").unwrap_or(".")).to_path_buf()
}

fn read_file(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn run_surface(terms: &[String]) -> Result<i32, String> {
    if terms.is_empty() {
        return Err("surface requires at least one search term".to_owned());
    }
    let ledger = root().join("docs/NEGATIVE_EVIDENCE.md");
    let rows = parse_rows(&read_file(&ledger)?);
    let terms: Vec<String> = terms.iter().map(|term| term.to_lowercase()).collect();
    let mut matched = 0usize;
    let mut binding = 0usize;
    let mut void = 0usize;

    for row in rows {
        let text = row.text();
        let lower = text.to_lowercase();
        if !terms.iter().all(|term| lower.contains(term)) {
            continue;
        }
        matched += 1;
        let label = match verdict(&row.header) {
            Verdict::Keep => {
                binding += 1;
                "BINDING KEEP"
            }
            Verdict::Reject
                if has_same_invocation_aa(&text)
                    || has_counted_mechanism(&text)
                    || has_profile_evidence(&text) =>
            {
                binding += 1;
                "BINDING REJECT"
            }
            Verdict::Reject => {
                void += 1;
                "VOID PRIOR"
            }
            Verdict::Other => "PRIOR INFO",
        };
        println!(
            "{label}: docs/NEGATIVE_EVIDENCE.md:{} — {}",
            row.line, row.header
        );
        println!("  retry predicate: {}", retry_predicate(&row));
    }

    if binding > 0 {
        println!(
            "BLOCKED — {binding} binding prior result(s), {void} void prior(s), \
             {matched} total match(es)."
        );
        return Ok(2);
    }
    if void > 0 {
        println!(
            "VOID PRIOR — {void} undecidable rejection(s) are non-binding; \
             {matched} total match(es)."
        );
        return Ok(0);
    }
    println!("CLEAR — {matched} informational match(es), no binding prior result.");
    Ok(0)
}

fn git_blob(spec: &str) -> Result<Option<String>, String> {
    let output = Command::new("git")
        .current_dir(root())
        .args(["show", spec])
        .output()
        .map_err(|error| format!("cannot run git show {spec}: {error}"))?;
    if output.status.success() {
        return String::from_utf8(output.stdout)
            .map(Some)
            .map_err(|error| format!("git blob {spec} is not UTF-8: {error}"));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("does not exist")
        || stderr.contains("exists on disk, but not in")
        || stderr.contains("Path '")
    {
        return Ok(None);
    }
    Err(format!("git show {spec} failed: {}", stderr.trim()))
}

fn run_validate_staged() -> Result<i32, String> {
    let mut violations = Vec::new();
    for path in LEDGER_PATHS {
        let staged = git_blob(&format!(":{path}"))?.unwrap_or_default();
        let head = git_blob(&format!("HEAD:{path}"))?.unwrap_or_default();
        violations.extend(validate_changed_text(&head, &staged, path));
    }
    for path in PUBLIC_DOC_PATHS {
        let staged = git_blob(&format!(":{path}"))?.unwrap_or_default();
        let head = git_blob(&format!("HEAD:{path}"))?.unwrap_or_default();
        violations.extend(validate_public_changed_text(&head, &staged, path));
    }
    if violations.is_empty() {
        println!(
            "CLEAR — staged rows satisfy decidable-REJECT, binary-SHA, result-class, \
             live-incumbent campaign-win, and public-claim contracts."
        );
        return Ok(0);
    }
    eprintln!("BLOCKED — staged ledger integrity violations:");
    for violation in violations {
        eprintln!("  {violation}");
    }
    Ok(2)
}

fn run_validate_entry(path: &str, line: usize) -> Result<i32, String> {
    let full_path = root().join(path);
    let rows = parse_rows(&read_file(&full_path)?);
    let row = rows
        .iter()
        .find(|row| row.line == line)
        .ok_or_else(|| format!("{path}:{line} is not the start of a `## ` ledger row"))?;
    if let Some(violation) = row_violation(row, path) {
        eprintln!("BLOCKED — {violation}");
        return Ok(2);
    }
    println!("CLEAR — {path}:{line} satisfies the ledger contract.");
    Ok(0)
}

fn usage() {
    eprintln!(
        "usage:\n  ledger_preflight surface <term> [more terms...]\n  \
         ledger_preflight validate-staged\n  \
         ledger_preflight validate-entry <ledger-path> <row-start-line>\n\n\
         `surface` also remains the default when the subcommand is omitted."
    );
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        usage();
        return Ok(64);
    };
    match command.as_str() {
        "surface" => run_surface(&args[1..]),
        "validate-staged" if args.len() == 1 => run_validate_staged(),
        "validate-entry" if args.len() == 3 => {
            let line = args[2]
                .parse()
                .map_err(|error| format!("invalid row-start-line `{}`: {error}", args[2]))?;
            run_validate_entry(&args[1], line)
        }
        "validate-staged" | "validate-entry" => {
            usage();
            Ok(64)
        }
        _ => run_surface(&args),
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("ledger_preflight: {error}");
            std::process::exit(70);
        }
    }
}
