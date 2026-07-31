#!/usr/bin/env python3
"""Incumbent-ratio coverage audit for franken_whisper.

Publishes three numbers, recomputed from the ledgers at whatever revision it is
run against:

    total perf KEEP claims held
    ...carrying a LIVE same-invocation vs-incumbent ratio
    ...not carrying one

Fleet policy under audit (README "Campaign wins use the actual incumbent"): a
competitive result requires the legacy incumbent to run side-by-side with
franken *in the same harness invocation*. A before/after comparison of franken
against itself is a maintenance self-speedup, never a competitive result.

Two counting traps this script exists to make un-repeatable:

1. **franken's own A/A interleaving matches "interleaved".** The campaign runs
   order-alternated franken-vs-franken nulls, so a bare `interleav` regex scores
   14 supported claims out of 44. Only tokens that *cannot* be emitted by a
   franken-vs-franken run count: `incumbent_bin_sha256`, `INCUMBENT_AB_`,
   `INCUMBENT-WIN`, `live incumbent`, `incumbent arm`.
2. **Non-claims look like claims.** `## `-headed sections include document
   structure ("Levers", "Result classes") that carry KEEP verdicts by quotation.
   A ledger row's title starts with its ISO date; structure does not.

Capability KEEPs (FEATURE / VALIDATION / ROBUSTNESS rows: quant formats,
`initial_prompt`, `--beam-size`, multilingual auto-detect) assert no ratio and
are reported separately. Counting them as "unsupported" would inflate the
failure in the opposite direction, which is its own dishonesty.

Usage:  python3 scripts/claim_coverage_audit.py [--detail]
"""
import re
import sys
from pathlib import Path

DOCS = ["docs/PERF_LEDGER.md", "docs/NEGATIVE_EVIDENCE.md"]

KEEP = re.compile(r"\b(KEEP|CAMPAIGN WIN|INCUMBENT-WIN)\b")
NOT_A_KEEP = re.compile(
    r"not a perf KEEP|BENCH-FREE|BLOCKED|NO VERDICT|UNMEASURED|REJECT|"
    r"do not retry|INVALID",
    re.I,
)
RATIO = re.compile(r"\d+\.\d+\s*[x×]|\bmedian\b|\bfaster\b|\bspeedup\b|RTF|ms\b", re.I)
CAPABILITY = re.compile(r"\bFEATURE\b|\bVALIDATION\b|\bROBUSTNESS\b", re.I)
BARE_RATIO = re.compile(r"\d+\.\d+\s*[x×]")
# A ledger row is dated; document structure is not.
LEDGER_ROW = re.compile(r"^\d{4}-\d{2}-\d{2}\b")

# Emitted only by a run that actually spawned the incumbent binary.
LIVE_INCUMBENT = re.compile(
    r"incumbent_bin_sha256|INCUMBENT_AB_|INCUMBENT-WIN|live[- ]incumbent|incumbent arm",
    re.I,
)
# Rows that quote the incumbent harness while explicitly declining to claim a
# campaign win against it.
SELF_DISCLAIMED = re.compile(r"NON-CAMPAIGN", re.I)

# Surfaces whisper.cpp does not implement at all: no incumbent arm can ever
# exist, so these are permanently unconvertible and the remedy is labelling.
NO_ARM_POSSIBLE = re.compile(
    r"router|Brier|loss[- ]matrix|adaptive|speculat|diariz|silhouette|cluster|"
    r"RunStore|SQL|storage|persist|SRT|VTT|NDJSON|robot|YouTube|paragraph|"
    r"request[- ]builder|CLI argument|stream|WindowManager|merge_segments|event",
    re.I,
)


def sections(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    title, body, out = None, [], []
    for line in lines:
        if line.startswith("## "):
            if title is not None:
                out.append((title, "\n".join(body)))
            title, body = line[3:].strip(), []
        else:
            body.append(line)
    if title is not None:
        out.append((title, "\n".join(body)))
    return out


def classify():
    claims, capability, excluded, structure = [], [], [], []
    for doc in DOCS:
        for title, body in sections(doc):
            blob = title + "\n" + body
            if not KEEP.search(blob):
                continue
            if not LEDGER_ROW.match(title):
                structure.append((doc, title))
                continue
            if NOT_A_KEEP.search(title):
                excluded.append((doc, title))
                continue
            if CAPABILITY.search(title) and not BARE_RATIO.search(title):
                capability.append((doc, title))
                continue
            if not RATIO.search(blob):
                capability.append((doc, title))
                continue
            supported = bool(LIVE_INCUMBENT.search(blob)) and not SELF_DISCLAIMED.search(
                title
            )
            convertible = not NO_ARM_POSSIBLE.search(title)
            claims.append((doc, title, supported, convertible))
    return claims, capability, excluded, structure


def main():
    detail = "--detail" in sys.argv
    claims, capability, excluded, structure = classify()
    supported = [c for c in claims if c[2]]
    unsupported = [c for c in claims if not c[2]]
    no_arm = [c for c in unsupported if not c[3]]
    convertible = [c for c in unsupported if c[3]]

    print(f"PERF_KEEP_CLAIMS_HELD={len(claims)}")
    print(f"  WITH_LIVE_SAME_INVOCATION_INCUMBENT_RATIO={len(supported)}")
    print(f"  WITHOUT={len(unsupported)}")
    print(f"    of which NO_INCUMBENT_ARM_CAN_EXIST={len(no_arm)}")
    print(f"    of which CONVERTIBLE_BUT_NOT_YET_MEASURED={len(convertible)}")
    print(f"CAPABILITY_KEEPS_NO_RATIO_OWED={len(capability)}")
    print(f"EXCLUDED_SELF_DISCLAIMED={len(excluded)}")
    print(f"DOC_STRUCTURE_NOT_CLAIMS={len(structure)}")

    if detail:
        for label, rows in (
            ("SUPPORTED", supported),
            ("UNSUPPORTED / NO ARM POSSIBLE", no_arm),
            ("UNSUPPORTED / CONVERTIBLE", convertible),
        ):
            print(f"\n--- {label} ({len(rows)}) ---")
            for _doc, title, _s, _c in rows:
                print(f"  {title[:150]}")


if __name__ == "__main__":
    main()
