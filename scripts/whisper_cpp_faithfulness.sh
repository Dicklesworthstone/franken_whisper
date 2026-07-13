#!/usr/bin/env bash
# whisper_cpp_faithfulness.sh — reproducible FAITHFULNESS comparison of
# franken_whisper native vs the original whisper.cpp `whisper-cli`.
#
# Companion to whisper_cpp_ab.sh (which measures SPEED). This measures OUTPUT
# QUALITY, the "faithful" half of "faster faithful whisper", the way it was
# validated in docs/PERF_FRONTIER.md ("Current-code headline" / faithfulness):
#
#   * WORD-AGREEMENT: longest-common-subsequence of the two normalized transcripts
#     (lowercase, punctuation-stripped) as a % of the longer — how closely native
#     tracks the reference (measured 88–98% on real speech; the gap is filler /
#     segmentation, NOT proper-noun or content errors).
#   * COVERAGE + LOOP guard: word counts and the most-repeated 5-gram per side —
#     catches the tiny.en TS content-drop (native short) and whisper.cpp's greedy
#     repetition loops on some clips (wc short + high 5-gram repeat).
#   * WER vs GROUND TRUTH (optional 4th `|`-field = a reference .txt): the rigorous
#     anchor. On jfk this is 0.0% for both models (native is exactly right).
#
# Native is forced to the real native engine (NATIVE_ROLLOUT_STAGE=sole) — WITHOUT
# it `fw transcribe` can route to the whisper.cpp backend and NOT test native.
# no-timestamps on both (the content-comparable path; note tiny.en *TS* drops ~50%
# unless FW_RETRY_FAILED_WINDOW=1, so compare in no_ts or set that flag).
#
# Env:
#   FW         franken_whisper `fw` binary   (default: $CARGO_TARGET_DIR/release/fw)
#   WC         whisper-cli binary            (default: legacy_whispercpp build)
#   MODEL_DIR  ggml model dir                (default: legacy_whispercpp models)
#   THREADS    whisper-cli threads           (default: 16 tiny / bump for turbo)
#
# Usage:  whisper_cpp_faithfulness.sh "MODEL.bin|CLIP.wav|LABEL[|GROUND_TRUTH.txt]" ...
#   e.g.  whisper_cpp_faithfulness.sh \
#           "ggml-tiny.en.bin|tests/fixtures/native/jfk.wav|jfk-tiny|jfk_gt.txt" \
#           "ggml-large-v3-turbo.bin|/tmp/track01.wav|track01-turbo"
# (whisper-cli can't read mp3 — make a wav first:
#  cargo run --release --example decode_to_wav -- in.mp3 out.wav)
set -u
CT=${CARGO_TARGET_DIR:-target}
FW=${FW:-$CT/release/fw}
WC=${WC:-legacy_whispercpp/whisper.cpp/build/bin/whisper-cli}
MODEL_DIR=${MODEL_DIR:-legacy_whispercpp/whisper.cpp/models}
THREADS=${THREADS:-16}
TMP=$(mktemp -d) || { echo "mktemp failed" >&2; exit 1; }
cleanup() { [ -n "${TMP:-}" ] && [ -d "$TMP" ] && rm -rf -- "$TMP"; }
trap cleanup EXIT

# metrics.py: prints "words_fw words_wc rep_fw rep_wc agree_pct [wer_fw wer_wc]"
metrics() { # $1=fw.txt $2=wc.txt [$3=ground_truth.txt]
  python3 - "$@" <<'PY'
import sys, re, difflib
def toks(p): return re.sub(r'[^a-z0-9 ]',' ',open(p).read().lower().replace('\n',' ')).split()
fw, wc = toks(sys.argv[1]), toks(sys.argv[2])
def maxrep(a):
    c={}
    for i in range(len(a)-4):
        k=tuple(a[i:i+5]); c[k]=c.get(k,0)+1
    return max(c.values()) if c else 0
lcs=sum(b.size for b in difflib.SequenceMatcher(None,fw,wc).get_matching_blocks())
agree=100.0*lcs/max(len(fw),len(wc),1)
out=[str(len(fw)),str(len(wc)),str(maxrep(fw)),str(maxrep(wc)),f'{agree:.1f}']
if len(sys.argv)>3:
    gt=toks(sys.argv[3])
    def wer(h):
        sm=difflib.SequenceMatcher(None,gt,h); s=d=i=0
        for t,i1,i2,j1,j2 in sm.get_opcodes():
            if t=='replace': s+=max(i2-i1,j2-j1)
            elif t=='delete': d+=i2-i1
            elif t=='insert': i+=j2-j1
        return 100.0*(s+d+i)/max(len(gt),1)
    out+=[f'{wer(fw):.1f}',f'{wer(wc):.1f}']
print(' '.join(out))
PY
}

echo "== franken_whisper native vs whisper.cpp FAITHFULNESS  (no_ts, threads=$THREADS) =="
echo "== $(date -u '+%Y-%m-%dT%H:%M:%SZ')  host=$(hostname)  load:$(uptime | grep -oE 'load average.*') =="
printf '%-22s %8s %8s %8s %8s %10s %s\n' LABEL fw_words wc_words fw_5rep wc_5rep agree% WER%_fw/wc
for spec in "$@"; do
  IFS='|' read -r model clip label gt <<< "$spec"
  mp="$MODEL_DIR/$model"
  FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole FRANKEN_WHISPER_MODEL_DIR="$MODEL_DIR" \
    FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL="$mp" \
    "$FW" transcribe --input "$clip" --no-timestamps --no-persist \
    >"$TMP/fw.txt" 2>/dev/null || { printf '%-22s FW FAILED\n' "$label"; continue; }
  "$WC" -m "$mp" -f "$clip" -t "$THREADS" -nt >"$TMP/wc.txt" 2>/dev/null \
    || { printf '%-22s WC FAILED\n' "$label"; continue; }
  read -r fww wcw fwr wcr agr werf werw < <(metrics "$TMP/fw.txt" "$TMP/wc.txt" ${gt:+"$gt"})
  printf '%-22s %8s %8s %8s %8s %9s%% %s\n' \
    "$label" "$fww" "$wcw" "$fwr" "$wcr" "$agr" "${werf:+$werf/$werw}"
done
echo "(fw_5rep/wc_5rep = most-repeated 5-gram = loop guard; a big count vs a low"
echo " word count = that side degraded. agree% = fw↔wc LCS. WER% only with a 4th"
echo " ground-truth field. See docs/PERF_FRONTIER.md 'Current-code headline'.)"
