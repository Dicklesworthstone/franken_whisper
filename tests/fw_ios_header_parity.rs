//! bd-simf: keep `fw-ios/include/fw_ios.h` in lockstep with `fw-ios/src/lib.rs`.
//!
//! The iOS header is hand-maintained while the C ABI lives in Rust. A symbol
//! added or renamed on one side without the other would ship broken Swift
//! bindings behind a fully green parent-suite CI, because nothing outside the
//! `fw-ios` crate reads that header. This test is the guard: it fails loudly,
//! naming the drifted symbols in both directions.
//!
//! Hermetic by construction: pure text parsing of two tracked files — no C
//! compiler, no Apple toolchain, no `fw-ios` build, no model weights.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn fw_ios_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fw-ios")
}

/// Remove `/* … */` and `// …` comments so doc-comment mentions like the
/// "Typical session" example (`fw_engine_open(…)`) cannot masquerade as
/// declarations.
fn strip_c_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Every lowercase `fw_*` identifier immediately followed by `(` — in a
/// comment-stripped declaration-only header these are exactly the declared
/// functions. (`FwEngine`, `FwProgressFn`, `FW_STREAM_LOAD` are excluded by
/// case.)
fn header_declarations(stripped: &str) -> BTreeSet<String> {
    let bytes = stripped.as_bytes();
    let mut symbols = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if stripped[i..].starts_with("fw_") && (i == 0 || !is_ident_byte(bytes[i - 1])) {
            let mut j = i;
            while j < bytes.len() && is_ident_byte(bytes[j]) {
                j += 1;
            }
            let name = &stripped[i..j];
            let mut k = j;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if k < bytes.len() && bytes[k] == b'(' {
                symbols.insert(name.to_owned());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    symbols
}

/// The exported ABI: a `#[unsafe(no_mangle)]` attribute (legacy `#[no_mangle]`
/// also accepted) attached to a `pub [unsafe] extern "C" fn fw_*` within the
/// next three lines. Requiring the attribute keeps non-exported
/// `extern "C"` helpers and callback type aliases out of the comparison.
fn rust_exports(lib_rs: &str) -> BTreeSet<String> {
    let lines: Vec<&str> = lib_rs.lines().collect();
    let mut symbols = BTreeSet::new();
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed != "#[unsafe(no_mangle)]" && trimmed != "#[no_mangle]" {
            continue;
        }
        for follower in lines.iter().skip(idx + 1).take(3) {
            if let Some(pos) = follower.find("extern \"C\" fn ") {
                let rest = follower[pos + "extern \"C\" fn ".len()..]
                    .trim_start()
                    .trim_start_matches("unsafe")
                    .trim_start();
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if name.starts_with("fw_") {
                    symbols.insert(name);
                }
                break;
            }
        }
    }
    symbols
}

fn load(rel: &str) -> String {
    let path = fw_ios_dir().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("bd-simf: cannot read {}: {error}", path.display()))
}

#[test]
fn fw_ios_header_matches_exported_abi() {
    let header = load("include/fw_ios.h");
    let lib_rs = load("src/lib.rs");

    let header_symbols = header_declarations(&strip_c_comments(&header));
    let export_symbols = rust_exports(&lib_rs);

    assert!(
        header_symbols.len() >= 10,
        "bd-simf non-vacuity: only {header_symbols_len} header symbols parsed — \
         the comment stripper or declaration scanner regressed",
        header_symbols_len = header_symbols.len()
    );
    assert!(
        export_symbols.len() >= 10,
        "bd-simf non-vacuity: only {} Rust exports parsed — the no_mangle \
         scanner regressed",
        export_symbols.len()
    );
    // Two anchors defining the ABI's shape; guards against a silent
    // double-failure where both scanners degrade to the same wrong subset.
    assert!(
        header_symbols.contains("fw_engine_open"),
        "header lost fw_engine_open"
    );
    assert!(
        export_symbols.contains("fw_string_free"),
        "lib.rs lost fw_string_free"
    );

    let missing_in_header: Vec<&String> = export_symbols.difference(&header_symbols).collect();
    let missing_in_rust: Vec<&String> = header_symbols.difference(&export_symbols).collect();

    assert!(
        missing_in_header.is_empty() && missing_in_rust.is_empty(),
        "bd-simf: fw_ios.h and fw-ios/src/lib.rs drifted.\n\
         Exported in Rust but MISSING from include/fw_ios.h (Swift side \
         cannot call these): {missing_in_header:?}\n\
         Declared in include/fw_ios.h but MISSING from src/lib.rs (header \
         promises an ABI that does not exist): {missing_in_rust:?}\n\
         Fix: add/rename the symbol on BOTH sides in the same commit."
    );
}

#[test]
fn drift_is_detected_when_the_header_loses_a_symbol() {
    // Planted negative: prove the comparator actually bites. Drop one
    // declared function from a synthetic header and require failure naming
    // exactly that symbol (RH-1: never trust a gate you have not seen fire).
    let header = load("include/fw_ios.h");
    let lib_rs = load("src/lib.rs");

    let stripped = strip_c_comments(&header);
    let full = header_declarations(&stripped);
    let victim = "fw_reset_cancel";
    assert!(
        full.contains(victim),
        "scanner lost the planted victim {victim}"
    );

    let without_victim: String =
        stripped.replace(&format!("{victim}("), "fw_removed_for_drill_down(");
    let mutated = header_declarations(&without_victim);

    let missing_in_header: Vec<&String> = full.difference(&mutated).collect();
    assert_eq!(missing_in_header, vec![&victim.to_owned()]);
    assert!(!rust_exports(&lib_rs).is_empty());
}
