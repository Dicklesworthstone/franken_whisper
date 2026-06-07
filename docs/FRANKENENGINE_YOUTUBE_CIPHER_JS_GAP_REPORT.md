# FrankenEngine — JS eval gaps blocking the YouTube cipher (and, later, BotGuard)

**Audience:** agents working on FrankenEngine.
**Author:** an agent porting yt-dlp's YouTube extractor to native Rust inside `franken_whisper` (`fw`), which wants to use FrankenEngine — *not* V8/QuickJS/boa — to run YouTube's player JavaScript.
**Date:** 2026-06-07.
**TL;DR:** FrankenEngine's public `HybridRouter::eval` runs literals, function-expression calls, and object-method dispatch, but is **missing the Array/String builtins and working loops** that the simplest real-world JS in our path needs. Below is a reproducible harness, a prioritized gap list with concrete pass/fail JS vectors, an acceptance gate, and one compile bug to fix. Closing the **P0** set unblocks YouTube *signature* deciphering; the **P2** set is the much larger lift toward BotGuard / PO-token attestation.

---

## 1. Why this matters / the end goal

`franken_whisper` is doing a clean-room, memory-safe, `#![forbid(unsafe_code)]` native-Rust port of yt-dlp's YouTube extractor, with **FrankenEngine as the JavaScript engine** (the user explicitly prefers it over boa). YouTube gates audio-stream URLs behind values computed by *obfuscated, frequently-rotated JavaScript* in the player's `base.js`:

1. **The signature cipher** (`s` param) — a `split → reverse/splice/swap → join`-style transform. **Comparatively simple JS.** This is the P0 target.
2. **The `n` throttling param** — similar shape plus arithmetic/bitwise. P1.
3. **BotGuard / PO-token attestation** — a heavily obfuscated VM (typed arrays, `Function` constructor, deep closures). This is among the most adversarial JS payloads in existence and is the P2/stretch target. (YouTube's 2025 SABR + PO-token rollout makes this necessary for media download — see §7.)

FrankenEngine is the ideal home for this: it's a native-Rust runtime *purpose-built for adversarial JS*, with no unsafe and no V8/QuickJS bindings. If it can run base.js's cipher (and eventually BotGuard), `fw` can extract YouTube audio with zero Python/JS-runtime dependencies. **That is currently blocked by the gaps below.**

---

## 2. Reproduction harness (standalone, ~2 min)

```toml
# /tmp/fe_eval_spike/Cargo.toml
[package]
name = "fe_eval_spike"
version = "0.0.0"
edition = "2024"
[dependencies]
# default-features = false is REQUIRED today — see §6 (the asupersync-integration
# default feature fails to compile against franken-decision 0.3.2).
frankenengine-engine = { path = "<...>/franken_engine/crates/franken-engine", default-features = false }
[workspace]
```

```rust
// /tmp/fe_eval_spike/src/main.rs
use frankenengine_engine::{HybridRouter, JsEngine};
fn run(label: &str, js: &str) {
    let mut e = HybridRouter::default();
    match e.eval(js) {
        Ok(o)  => println!("{label:34} OK  -> {:?}", o.value),
        Err(e) => println!("{label:34} ERR -> {}", e.message),
    }
}
fn main() {
    // --- confirmed WORKING today ---
    run("literal",              "42");                                   // OK -> "42"
    run("function-expr call",   "var f=function(x){return x>2}; f(3)");  // OK -> "true"
    run("object method dispatch","var o={f:function(x){return x+1}}; o.f(5)"); // OK -> "6"
    // --- the P0 GAPS (all currently fail) ---
    run("array.reverse+join",   r#"var a=["1","2","3"]; a.reverse(); a.join("")"#);
    run("array.splice",         r#"var a=["1","2","3","4"]; a.splice(0,2); a.join("")"#);
    run("array.slice",          r#"var a=["1","2","3"]; a.slice(1).join("")"#);
    run("string.split",         r#"var s="abc"; s.split("").join("-")"#);
    run("String.fromCharCode",  r#"String.fromCharCode(72,105)"#);
    run("charCodeAt",           r#"var x="A"; x.charCodeAt(0)"#);
    run("for loop (5 iters)",   r#"var s=0; for(var i=0;i<5;i++){s+=i;} s"#);
    run("element swap",         r#"var a=["1","2","3"]; var c=a[0]; a[0]=a[2]; a[2]=c; a.join("")"#);
}
```

Build/run reusing FrankenEngine's own target dir so it links fast:
```bash
cd /tmp/fe_eval_spike
CARGO_TARGET_DIR=<...>/franken_engine/target cargo run --quiet
```

---

## 3. What works today (keep it working — these are the regression anchors)

| JS | Result | Status |
|---|---|---|
| `42` | `"42"` | ✅ |
| `var f=function(x){return x>2}; f(3)` | `"true"` | ✅ |
| `var o={f:function(x){return x+1}}; o.f(5)` | `"6"` | ✅ object-property function dispatch |

Object-method dispatch working is the bright spot — the cipher's helper-object pattern (`Mt.rv(a)`) relies on it.

---

## 4. P0 gaps — REQUIRED to run the YouTube signature cipher

Each row: minimal JS, **observed** behavior today, **expected** output. These are not exotic — they are core ECMAScript that every cipher uses. Treat each as a unit test (input → expected) in your suite.

| # | JS | Observed today | Expected |
|---|---|---|---|
| P0-1 | `var a=["1","2","3"]; a.reverse(); a.join("")` | `ERR type error: expected function, got undefined` | `"321"` |
| P0-2 | `var a=["1","2","3","4"]; a.splice(0,2); a.join("")` | same ERR | `"34"` |
| P0-3 | `var a=["1","2","3"]; a.slice(1).join("")` | same ERR | `"23"` |
| P0-4 | `var s="abc"; s.split("").join("-")` | same ERR | `"a-b-c"` |
| P0-5 | `String.fromCharCode(72,105)` | `ERR type error: expected object, got undefined` | `"Hi"` |
| P0-6 | `var x="A"; x.charCodeAt(0)` | (`String.prototype.charCodeAt` missing) | `"65"` |
| P0-7 | `var s=0; for(var i=0;i<5;i++){s+=i;} s` | **`ERR instruction budget exhausted: 100000/100000`** | `"10"` |
| P0-8 | `var a=["1","2","3"]; var c=a[0]; a[0]=a[2]; a[2]=c; a.join("")` | ERR (the `.join`) | `"321"` |

### The four concrete work items behind P0

**(a) `Array.prototype` methods.** Implement `reverse()` (in-place), `splice(start, deleteCount)` (in-place removal — the cipher uses `splice(0, b)`), `slice(start[, end])` (copy), `join(sep)`, and confirm element read/assign `a[i]` / `a[i]=x` and `a.length`. These are the literal building blocks of the signature transform. (`indexOf`/`lastIndexOf`/`concat`/`fill` are P1.)

**(b) `String` builtins.** `String.prototype.split(sep)` (especially `split("")` → array of single-char strings, and split on a separator), `String.fromCharCode(...codes)` (static on the `String` global), `String.prototype.charCodeAt(i)`. (`slice`/`substr`/`substring`/`replace`/`indexOf` are P1.)

**(c) Loops + instruction budget.** A **5-iteration `for` loop exhausts the 100 000-instruction budget** — this is a smoking gun that either loop execution is mis-compiled (each iteration costs absurd/unbounded instructions) or the default budget is far too low *and* not raisable through the public API. Fix loop execution, and expose a way to **set/raise the instruction budget** via the embedding API (cipher functions legitimately loop over hundreds of chars; BotGuard will need millions). `while`/`do-while` should follow the same fix.

**(d) Parser: leading string-literal statement.** A program that *starts* with a string-literal expression (e.g. `"abc".split("")`) currently errors with `UnsupportedSyntax: unterminated or malformed string literal (line 1, column 1)`. A leading string-literal *expression statement* is valid JS (it's not a directive prologue once followed by `.member`). Minified base.js routinely starts expressions with string/array literals. (Workaround on our side: wrap in `var x=...;` — but please fix, because we feed extracted snippets verbatim.)

---

## 5. Acceptance gate for "signature cipher unblocked"

When `HybridRouter::eval` returns the expected value for **all of §4** *and* for this representative, self-contained YouTube-signature-shaped function, the cipher path is unblocked:

```js
var Mt = {
  rv: function(a){ a.reverse(); },
  sp: function(a,b){ a.splice(0,b); },
  sw: function(a,b){ var c=a[0]; a[0]=a[b%a.length]; a[b%a.length]=c; }
};
function decipherSig(a){
  a = a.split("");
  Mt.sw(a,3); Mt.rv(a,0); Mt.sp(a,2); Mt.sw(a,1); Mt.rv(a,0);
  return a.join("");
}
decipherSig("0123456789")
```
**Ground truth (hand-computed): `"31204576"`.**
This mirrors the exact structure yt-dlp extracts from base.js (a helper object of `reverse`/`splice`/`swap` functions, applied in a sequence) — it exercises split, object-method dispatch, in-place array mutation via helper functions, modulo indexing, and join together. Getting `"31204576"` from it is the milestone.

**Stretch acceptance (the real thing):** `fw` will hand you the *actual* signature function + its helper object extracted from a live `base.js` (a few hundred lines of minified JS) plus a real encrypted `s` value; producing the same deciphered output a reference engine produces is the true end-to-end proof. We can supply that fixture on request.

---

## 6. Compile bug to fix (independent of the eval gaps)

`frankenengine-engine`'s **default feature `asupersync-integration`** (which pulls `dep:franken-decision`) **fails to compile against `franken-decision` 0.3.2**:

```
error[E0053]: method `update_posterior` has an incompatible type for trait
  --> (decision integration code)
error[E0308]: mismatched types  (tee_live_quote.rs)
```

`franken-decision` 0.3.2 changed `DecisionContract::update_posterior` to return `Result<(), UpdatePosteriorError>` (and `evaluate()` to return `Result<DecisionOutcome, ValidationError>`). FrankenEngine's code still uses the 0.3.1 signatures. Today a fresh consumer must use `default-features = false` (or pin `franken-decision = "=0.3.1"`) to build at all. **Please migrate to the 0.3.2 API** — wire `update_posterior`/`evaluate` errors into a deterministic fallback (that's exactly what `franken_whisper` did in its own 0.3.2 migration; happy to share the diff). Until then, document `default-features = false` as the supported embedding configuration.

(Note: `default-features = false` is fine for our use — we want only the JS engine, not the decision/evidence control plane. But the default-feature path being red is a footgun for any external consumer.)

---

## 7. P1 / P2 — the road past the simple cipher (so you can see the trajectory)

**P1 — needed for the `n` throttling transform + obfuscation robustness:**
- Bitwise ops `& | ^ << >> >>>` with correct 32-bit semantics; modulo on the same.
- `String.prototype.slice/substr/substring/replace/indexOf`; `Array.prototype.indexOf/lastIndexOf/concat/push/pop/shift/unshift`.
- `typeof`, ternary `?:`, the comma operator (minifiers emit it constantly), `while`/`do-while`, `switch`.
- `Math.floor/abs/imul`, parseInt/parseFloat, Number/String coercions matching ECMAScript.

**P2 — needed for BotGuard / PO-token attestation (the hard, high-value target):**
- **Typed arrays** (`Uint8Array`, `Int32Array`, `DataView`, `ArrayBuffer`) — BotGuard is built on these.
- **The `Function` constructor / dynamic code generation** — BotGuard builds and runs code at runtime. (This is exactly the adversarial-JS scenario FrankenEngine was designed for — and where its IFC/sandbox model is a real asset over a naive engine.)
- `try/catch/finally`, `RegExp` (used in extraction *and* inside BotGuard), fuller `JSON`, `Date`/`performance` shims (deterministic/sandboxed), `Object`/`Array` statics, closures over mutable captured state, getters/setters.
- A **large, configurable instruction/time budget** (BotGuard runs are heavy) plus the determinism guarantees FrankenEngine already prizes.

Reaching P2 is a significant ECMAScript-coverage effort, but it is the thing that would let FrankenEngine do something almost nothing else can: **generate YouTube PO tokens natively, in memory-safe Rust, with no browser and no external JS runtime.** That's a genuinely differentiating capability.

---

## 8. Suggested ordering for FrankenEngine

1. Fix the **for-loop / instruction-budget** bug (P0-7) — without working loops nothing else matters; the budget gate suggests a deeper interpreter issue worth root-causing first.
2. Land **`Array.prototype` reverse/splice/slice/join** + element read/assign (P0-1,2,3,8).
3. Land **`String.split` + `String.fromCharCode` + `charCodeAt`** (P0-4,5,6).
4. Fix the **leading-string-literal parser** case (P0-d).
5. Make the **§5 acceptance function** return `"31204576"`; ping `franken_whisper` for the live base.js fixture (stretch).
6. Fix the **franken-decision 0.3.2** compile (§6) so the default-feature build is green.
7. Then P1 (`n` transform), then P2 (BotGuard) as separate epics.

Reproduce everything with the §2 harness. When §4 + §5 pass, `fw`'s native signature deciphering is unblocked and we resume the port on the `franken_whisper` side.
