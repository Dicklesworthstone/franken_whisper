# FrankenEngine — JS eval status for the YouTube cipher + BotGuard

**Audience:** agents working on FrankenEngine.
**Author:** an agent porting yt-dlp's YouTube extractor to native Rust inside `franken_whisper` (`fw`), which wants to use FrankenEngine — *not* V8/QuickJS/boa — to run YouTube's player JavaScript.
**Date:** 2026-06-07. **Tested against:** `main` @ `8ca80bfc`.

> **CORRECTION NOTICE.** An earlier version of this report (same date) claimed
> FrankenEngine was missing Array/String builtins and working loops. That was
> **wrong** — it was run against a stale `v0.1.0` checkout (the clone was detached
> on the release tag, 228 commits behind `main`). Re-tested against `main`
> @ `8ca80bfc`: **the YouTube signature cipher runs correctly today** (the
> acceptance gate below passes). The only remaining gaps are three advanced
> features needed for *BotGuard / PO-token* attestation. This report has been
> rewritten to reflect `main`.

**TL;DR:** On `main`, FrankenEngine's public `HybridRouter::eval` **already runs
the YouTube signature cipher** — array methods, `String.split/fromCharCode/charCodeAt`,
loops, bitwise, `RegExp`, `String.replace`, `Array.map`+closures, `JSON`,
`Math.imul` all work. The acceptance gate (`decipherSig("0123456789") ==
"31204576"`) passes. **Three features remain** — typed arrays, the `Function`
constructor, and `try/catch` — and they block only the *much harder* BotGuard /
PO-token path, not the cipher.

---

## 1. Why this matters

`franken_whisper` is doing a clean-room, memory-safe, `#![forbid(unsafe_code)]`
native-Rust port of yt-dlp's YouTube extractor, with **FrankenEngine as the
JavaScript engine** (the user explicitly prefers it over boa). YouTube gates
audio behind JS in the player's `base.js`:

1. **The signature cipher** (`s` param) — `split → reverse/splice/swap → join`.
   **✅ FrankenEngine `main` runs this today** (see §3).
2. **The `n` throttling param** — cipher-shaped plus arithmetic/bitwise. Bitwise,
   loops, and `charCodeAt`/`fromCharCode` all work, so this is **very likely
   covered** (pending a real-base.js test).
3. **BotGuard / PO-token attestation** — a heavily obfuscated VM. **Blocked** on
   the three features in §4. (YouTube's 2025 SABR + PO-token rollout makes this
   necessary for media *download*.)

FrankenEngine running adversarial JS natively (no V8/QuickJS, no unsafe) is
exactly what makes a zero-Python YouTube extractor possible. The cipher is done;
BotGuard is the remaining JS lift.

---

## 2. Reproduction harness (standalone, ~2 min)

```toml
# /tmp/fe_eval_spike/Cargo.toml
[package]
name = "fe_eval_spike"
version = "0.0.0"
edition = "2024"
[dependencies]
# default-features = false avoids the asupersync-integration feature (which pulls
# franken-decision); the JS engine is all we need. (NB: on a fresh checkout the
# default-feature build's franken-decision dep may need a version that matches
# main's API — default-features=false sidesteps it.)
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
fn main() { /* the §3 + §4 vectors below */ }
```
```bash
cd /tmp/fe_eval_spike
CARGO_TARGET_DIR=<...>/franken_engine/target cargo run --quiet
```

---

## 3. Confirmed WORKING on `main` @ 8ca80bfc (cipher unblocked) — keep these as regression anchors

| JS | Result | node ground truth |
|---|---|---|
| `var a=["1","2","3"]; a.reverse(); a.join("")` | `"321"` | `"321"` ✅ |
| `var a=["1","2","3","4"]; a.splice(0,2); a.join("")` | `"34"` | `"34"` ✅ |
| `var a=["1","2","3"]; a.slice(1).join("")` | `"23"` | `"23"` ✅ |
| `var s="abc"; s.split("").join("-")` | `"a-b-c"` | `"a-b-c"` ✅ |
| `String.fromCharCode(66)` | `"B"` | `"B"` ✅ |
| `var x="A"; x.charCodeAt(0)` | `"65"` | `"65"` ✅ |
| `var s=0; for(var i=0;i<5;i++){s+=i;} s` | `"10"` | `10` ✅ |
| `/ab+c/.test("xabbbcx")` | `"true"` | `true` ✅ |
| `"a1b2c3".replace(/[0-9]/g,"_")` | `"a_b_c_"` | `"a_b_c_"` ✅ |
| `[1,2,3].map(function(x){return x*x;}).join(",")` | `"1,4,9"` | `"1,4,9"` ✅ |
| `JSON.stringify(JSON.parse('{"a":1,"b":[2,3]}'))` | `{"a":1,"b":[2,3]}` | ✅ |
| `(0xFFFFFFFF & 0x0F) >>> 0` | `"15"` | `15` ✅ |
| `Math.floor(Math.imul(7,7)/2)` | `"24"` | `24` ✅ |

### Acceptance gate — PASSES on `main`

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
decipherSig("0123456789")   // -> "31204576"  (node ground truth: "31204576") ✅
```
This mirrors the exact structure yt-dlp extracts from base.js (helper object of
`reverse`/`splice`/`swap`, applied in sequence). **It returns `"31204576"` on
`main` today.** Please keep it as a pinned regression test. (Stretch: `fw` can
supply a real base.js-extracted function + encrypted `s` for an end-to-end check.)

---

## 4. Remaining gaps on `main` — these block BotGuard / PO-token only (not the cipher)

Each row: minimal JS, observed on `main`, expected. BotGuard is a typed-array VM
that builds and runs code at runtime and uses exceptions for control flow, so all
three are load-bearing for PO-token generation.

| # | JS | Observed on `main` | Expected |
|---|---|---|---|
| G-1 typed arrays | `var a=new Uint8Array(3); a[0]=255; a[1]=1; a[0]+a[1]` | `ERR type error: expected function, got undefined` | `"256"` |
| G-2 `Function` ctor | `var f=new Function("x","return x*2"); f(21)` | `ERR type error: expected function, got undefined` | `"42"` |
| G-3 `try/catch` | `var r; try{ null.x; }catch(e){ r="caught"; } r` | `ERR type error: expected object, got null` (the throw escapes the `catch`) | `"caught"` |

### The three work items

**(G-1) Typed arrays** — `Uint8Array`/`Int32Array`/`Uint32Array`/`DataView`/`ArrayBuffer`
constructors + indexed get/set with the correct wrap/clamp semantics. BotGuard is
built on these.

**(G-2) The `Function` constructor / dynamic code generation** — `new Function(args, body)`
must compile and return a callable. BotGuard generates and runs code at runtime.
This is exactly the adversarial-JS scenario FrankenEngine's IFC/sandbox model is
designed for — running attacker-built code under containment is the differentiator.

**(G-3) `try/catch/finally`** — exceptions thrown inside a `try` must be caught by
the `catch` block (today the throw escapes to the eval boundary). BotGuard uses
exceptions for control flow and anti-tamper checks.

Likely also needed for BotGuard (untested on `main`, worth confirming): a large /
configurable instruction budget (BotGuard runs are heavy), `Date`/`performance`
shims (deterministic/sandboxed), and broader `Object` statics.

---

## 5. Net status & ask

- **Signature cipher: DONE on `main`.** No action needed beyond pinning the §3
  acceptance gate as a regression test. `fw`'s native cipher path is unblocked.
- **BotGuard / PO-token: blocked on G-1/G-2/G-3** (typed arrays, `Function` ctor,
  `try/catch`). Landing those three is what would let FrankenEngine mint YouTube
  PO tokens natively in memory-safe Rust with no browser and no external JS
  runtime — a genuinely differentiating capability.

(Earlier-reported "missing array methods / broken loops / compile bug" items were
artifacts of the stale `v0.1.0` checkout and are **resolved on `main`** — please
disregard them.)
