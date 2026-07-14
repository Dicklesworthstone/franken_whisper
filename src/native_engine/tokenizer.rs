//! Whisper token decoding and special-token map (pure Rust).
//!
//! This is a faithful port of whisper.cpp's vocabulary handling. The most
//! safety-critical part is the *derivation of the special-token ids* from the
//! vocabulary size: whisper.cpp hard-codes a set of base ids for the
//! English-only layout and then shifts them by a vocab-size-dependent delta for
//! multilingual models. We replicate that arithmetic exactly rather than
//! guessing ids, so the map agrees bit-for-bit with whisper.cpp for every model
//! family (see [`Tokenizer::from_vocab`] for the cited line numbers).
//!
//! ## Decoding
//!
//! Whisper tokens are raw byte strings (GPT-2 byte-level BPE pieces). A single
//! UTF-8 character can be split across two tokens, so [`Tokenizer::decode`]
//! concatenates the raw bytes of every token first and performs a *single*
//! lossy UTF-8 conversion at the end. Converting per-token would mangle
//! multi-byte characters that straddle a token boundary.
//!
//! ## Timestamps
//!
//! Token ids `>= timestamp_begin` encode a time offset of `0.02 s` per step
//! from the start of the current 30-second window
//! (`time = (id - timestamp_begin) * 0.02`).

use super::WhisperHParams;
use std::collections::HashSet;

/// Language list ported verbatim from whisper.cpp's `g_lang` map
/// (`whisper.cpp` lines 280-381). Each entry is `(code, id, english_name)`.
///
/// The language *id* is the index used in the prompt token layout: the model's
/// language token for id `n` is `sot + 1 + n`. The codes appear here in `id`
/// order, which is the order whisper expects.
#[rustfmt::skip]
pub const LANGUAGES: &[(&str, i32, &str)] = &[
    ("en",  0,  "english"),        ("zh",  1,  "chinese"),
    ("de",  2,  "german"),         ("es",  3,  "spanish"),
    ("ru",  4,  "russian"),        ("ko",  5,  "korean"),
    ("fr",  6,  "french"),         ("ja",  7,  "japanese"),
    ("pt",  8,  "portuguese"),     ("tr",  9,  "turkish"),
    ("pl",  10, "polish"),         ("ca",  11, "catalan"),
    ("nl",  12, "dutch"),          ("ar",  13, "arabic"),
    ("sv",  14, "swedish"),        ("it",  15, "italian"),
    ("id",  16, "indonesian"),     ("hi",  17, "hindi"),
    ("fi",  18, "finnish"),        ("vi",  19, "vietnamese"),
    ("he",  20, "hebrew"),         ("uk",  21, "ukrainian"),
    ("el",  22, "greek"),          ("ms",  23, "malay"),
    ("cs",  24, "czech"),          ("ro",  25, "romanian"),
    ("da",  26, "danish"),         ("hu",  27, "hungarian"),
    ("ta",  28, "tamil"),          ("no",  29, "norwegian"),
    ("th",  30, "thai"),           ("ur",  31, "urdu"),
    ("hr",  32, "croatian"),       ("bg",  33, "bulgarian"),
    ("lt",  34, "lithuanian"),     ("la",  35, "latin"),
    ("mi",  36, "maori"),          ("ml",  37, "malayalam"),
    ("cy",  38, "welsh"),          ("sk",  39, "slovak"),
    ("te",  40, "telugu"),         ("fa",  41, "persian"),
    ("lv",  42, "latvian"),        ("bn",  43, "bengali"),
    ("sr",  44, "serbian"),        ("az",  45, "azerbaijani"),
    ("sl",  46, "slovenian"),      ("kn",  47, "kannada"),
    ("et",  48, "estonian"),       ("mk",  49, "macedonian"),
    ("br",  50, "breton"),         ("eu",  51, "basque"),
    ("is",  52, "icelandic"),      ("hy",  53, "armenian"),
    ("ne",  54, "nepali"),         ("mn",  55, "mongolian"),
    ("bs",  56, "bosnian"),        ("kk",  57, "kazakh"),
    ("sq",  58, "albanian"),       ("sw",  59, "swahili"),
    ("gl",  60, "galician"),       ("mr",  61, "marathi"),
    ("pa",  62, "punjabi"),        ("si",  63, "sinhala"),
    ("km",  64, "khmer"),          ("sn",  65, "shona"),
    ("yo",  66, "yoruba"),         ("so",  67, "somali"),
    ("af",  68, "afrikaans"),      ("oc",  69, "occitan"),
    ("ka",  70, "georgian"),       ("be",  71, "belarusian"),
    ("tg",  72, "tajik"),          ("sd",  73, "sindhi"),
    ("gu",  74, "gujarati"),       ("am",  75, "amharic"),
    ("yi",  76, "yiddish"),        ("lo",  77, "lao"),
    ("uz",  78, "uzbek"),          ("fo",  79, "faroese"),
    ("ht",  80, "haitian creole"), ("ps",  81, "pashto"),
    ("tk",  82, "turkmen"),        ("nn",  83, "nynorsk"),
    ("mt",  84, "maltese"),        ("sa",  85, "sanskrit"),
    ("lb",  86, "luxembourgish"),  ("my",  87, "myanmar"),
    ("bo",  88, "tibetan"),        ("tl",  89, "tagalog"),
    ("mg",  90, "malagasy"),       ("as",  91, "assamese"),
    ("tt",  92, "tatar"),          ("haw", 93, "hawaiian"),
    ("ln",  94, "lingala"),        ("ha",  95, "hausa"),
    ("ba",  96, "bashkir"),        ("jw",  97, "javanese"),
    ("su",  98, "sundanese"),      ("yue", 99, "cantonese"),
];

/// Non-speech symbol strings, ported verbatim from whisper.cpp's
/// `non_speech_tokens` const (`whisper.cpp` lines 6131-6136). These are matched
/// against the vocabulary (both bare and `' '`-prefixed) to build the suppress
/// set; see [`Tokenizer::build_non_speech`].
#[rustfmt::skip]
const NON_SPEECH_SYMBOLS: &[&str] = &[
    "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^",
    "_", "`", "{", "|", "}", "~", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--",
    "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪",
    "♪♪♪", "♩", "♪", "♫", "♬", "♭", "♮", "♯",
];

/// Seconds of audio represented by one timestamp-token step (whisper uses
/// `0.02 s`, i.e. 20 ms, per timestamp granularity).
pub const TIMESTAMP_STEP_SEC: f32 = 0.02;

/// Whisper tokenizer: raw-byte vocabulary plus the derived special-token map.
///
/// Construct with [`Tokenizer::from_vocab`]. All special-id fields are derived
/// from `hparams.n_vocab` following whisper.cpp's arithmetic and exposed both as
/// public fields and (for ergonomics) via getter methods.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Raw byte string for every token id, indexed by id.
    tokens: Vec<Vec<u8>>,
    /// `n_vocab` as declared in the model header (may exceed `tokens.len()`
    /// when the file omits the trailing special tokens; whisper.cpp synthesises
    /// the remainder).
    n_vocab: i32,
    /// Whether this is a multilingual model (`n_vocab >= 51865`).
    multilingual: bool,
    /// Number of language tokens for this model family.
    n_langs: i32,

    /// End-of-transcript token.
    pub eot: i32,
    /// Start-of-transcript token.
    pub sot: i32,
    /// Translate task token (multilingual only).
    pub translate: i32,
    /// Transcribe task token (multilingual only).
    pub transcribe: i32,
    /// `[TDRZ]` speaker-turn token used by tinydiarize models.
    pub solm: i32,
    /// Start-of-previous-context token (`sot_prev`).
    pub sot_prev: i32,
    /// No-speech token (`nosp`).
    pub no_speech: i32,
    /// No-timestamps token (`not`).
    pub no_timestamps: i32,
    /// First timestamp token (`beg`); timestamp ids are `>= this`.
    pub timestamp_begin: i32,

    /// Suppress set: ids of non-speech symbol tokens present in this vocab.
    non_speech: Vec<i32>,
}

impl Tokenizer {
    /// Build a tokenizer from the model hyper-parameters and the raw per-id
    /// token byte strings (as read from the ggml vocab section).
    ///
    /// ## Special-id derivation (ported from whisper.cpp)
    ///
    /// whisper.cpp declares the English-only base ids in the `whisper_vocab`
    /// struct (`whisper.cpp` lines 439-449):
    ///
    /// ```text
    /// token_eot        = 50256;
    /// token_sot        = 50257;
    /// token_translate  = 50357;
    /// token_transcribe = 50358;
    /// token_solm       = 50359;
    /// token_prev       = 50360;
    /// token_nosp       = 50361;
    /// token_not        = 50362;
    /// token_beg        = 50363;
    /// ```
    ///
    /// For multilingual models it then applies (`whisper.cpp` lines 1624-1639):
    ///
    /// ```text
    /// if (is_multilingual()) {            // n_vocab >= 51865
    ///     token_eot++;                    // +1
    ///     token_sot++;                    // +1
    ///     int dt = num_languages() - 98;  // num_languages = n_vocab - 51765 - 1
    ///     token_translate  += dt;
    ///     token_transcribe += dt;
    ///     token_solm       += dt;
    ///     token_prev       += dt;
    ///     token_nosp       += dt;
    ///     token_not        += dt;
    ///     token_beg        += dt;
    /// }
    /// ```
    ///
    /// The language token for language id `n` is `sot + 1 + n`
    /// (`whisper.cpp` `whisper_token_lang`, line 4247-4248).
    #[must_use]
    pub fn from_vocab(hparams: &WhisperHParams, vocab_tokens: Vec<Vec<u8>>) -> Self {
        let n_vocab = hparams.n_vocab;
        let multilingual = hparams.is_multilingual();

        // whisper.cpp lines 439-449: English-only base ids.
        let mut eot = 50256;
        let mut sot = 50257;
        let mut translate = 50357;
        let mut transcribe = 50358;
        let mut solm = 50359;
        let mut sot_prev = 50360;
        let mut no_speech = 50361;
        let mut no_timestamps = 50362;
        let mut timestamp_begin = 50363;

        // whisper.cpp line 456: num_languages() = n_vocab - 51765 - (multilingual ? 1 : 0).
        let n_langs = n_vocab - 51765 - i32::from(multilingual);

        if multilingual {
            // whisper.cpp lines 1626-1638.
            eot += 1;
            sot += 1;
            let dt = n_langs - 98;
            translate += dt;
            transcribe += dt;
            solm += dt;
            sot_prev += dt;
            no_speech += dt;
            no_timestamps += dt;
            timestamp_begin += dt;
        }

        let mut tk = Self {
            tokens: vocab_tokens,
            n_vocab,
            multilingual,
            n_langs,
            eot,
            sot,
            translate,
            transcribe,
            solm,
            sot_prev,
            no_speech,
            no_timestamps,
            timestamp_begin,
            non_speech: Vec::new(),
        };
        tk.non_speech = tk.build_non_speech();
        tk
    }

    /// Build the non-speech suppress set by scanning the vocab for each symbol
    /// in [`NON_SPEECH_SYMBOLS`], both bare and `' '`-prefixed, plus the special
    /// `" -"` and `" '"` cases. Ported from whisper.cpp lines 6279-6295.
    fn build_non_speech(&self) -> Vec<i32> {
        // Index the small suppress-pattern side once, then walk the large
        // vocabulary once.  The previous pattern-major search restarted a
        // linear 50k-token scan for every bare/prefixed symbol (~130 scans).
        let mut remaining = HashSet::with_capacity(NON_SPEECH_SYMBOLS.len() * 2 + 2);
        for sym in NON_SPEECH_SYMBOLS {
            // whisper.cpp line 6281: { token, " " + token }.
            remaining.insert(sym.as_bytes().to_vec());
            let mut prefixed = Vec::with_capacity(sym.len() + 1);
            prefixed.push(b' ');
            prefixed.extend_from_slice(sym.as_bytes());
            remaining.insert(prefixed);
        }
        // whisper.cpp lines 6290-6294: also suppress " -" and " '".
        remaining.insert(b" -".to_vec());
        remaining.insert(b" '".to_vec());

        let mut ids = Vec::with_capacity(remaining.len());
        for (id, token) in self.tokens.iter().enumerate() {
            // Removing a match preserves the old first-id semantics when a
            // malformed vocabulary contains duplicate byte strings.
            if remaining.remove(token.as_slice())
                && let Ok(id) = i32::try_from(id)
            {
                ids.push(id);
            }
            if remaining.is_empty() {
                break;
            }
        }
        ids
    }

    /// Declared vocabulary size (`n_vocab` from the model header).
    #[must_use]
    pub fn vocab_size(&self) -> i32 {
        self.n_vocab
    }

    /// Whether this is a multilingual model.
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Number of language tokens for this model family (0 for English-only in
    /// practice, since English-only models carry no language tokens; the value
    /// still follows whisper.cpp's formula).
    #[must_use]
    pub fn num_languages(&self) -> i32 {
        self.n_langs
    }

    /// Raw bytes of token `id`, or `None` if out of range.
    #[must_use]
    pub fn token_bytes(&self, id: i32) -> Option<&[u8]> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .map(Vec::as_slice)
    }

    /// Whether `id` is a special (non-text) token: `eot`, `sot`, any language
    /// token, the task/control tokens, or a timestamp token.
    #[must_use]
    pub fn is_special(&self, id: i32) -> bool {
        id >= self.eot
    }

    /// Whether `id` is a timestamp token (`id >= timestamp_begin`).
    #[must_use]
    pub fn is_timestamp(&self, id: i32) -> bool {
        id >= self.timestamp_begin
    }

    /// Time offset in seconds encoded by timestamp token `id`
    /// (`(id - timestamp_begin) * 0.02`). For non-timestamp ids this returns a
    /// negative value; callers should gate on [`Tokenizer::is_timestamp`].
    #[must_use]
    pub fn timestamp_sec(&self, id: i32) -> f32 {
        (id - self.timestamp_begin) as f32 * TIMESTAMP_STEP_SEC
    }

    /// Language token id for a code (e.g. `"en"`), or `None` if the code is
    /// unknown or this model has no language tokens (English-only).
    ///
    /// Per whisper.cpp `whisper_token_lang` (line 4247): `sot + 1 + lang_id`.
    #[must_use]
    pub fn language_token(&self, lang_code: &str) -> Option<i32> {
        if !self.multilingual {
            return None;
        }
        let lang_id = LANGUAGES
            .iter()
            .find(|(code, _, _)| *code == lang_code)
            .map(|(_, id, _)| *id)?;
        if lang_id >= self.n_langs {
            return None;
        }
        Some(self.sot + 1 + lang_id)
    }

    /// The non-speech suppress set (sorted token ids).
    #[must_use]
    pub fn non_speech_tokens(&self) -> &[i32] {
        &self.non_speech
    }

    /// Build the start-of-transcript prompt sequence.
    ///
    /// Ported from whisper.cpp `prompt_init` construction (lines 6975-6999):
    /// - always starts with `sot`;
    /// - multilingual models append the language token then the task token
    ///   (`translate` or `transcribe`);
    /// - English-only models carry NO language/task tokens;
    /// - if `timestamps` is false, the `no_timestamps` token is appended.
    ///
    /// `language` defaults to `"en"` when `None`. Unknown language codes fall
    /// back to `"en"` (mirroring whisper.cpp's `whisper_lang_id` default).
    #[must_use]
    pub fn sot_sequence(
        &self,
        language: Option<&str>,
        translate: bool,
        timestamps: bool,
    ) -> Vec<i32> {
        let mut seq = vec![self.sot];
        if self.multilingual {
            let lang = language.unwrap_or("en");
            let lang_tok = self.language_token(lang).unwrap_or_else(|| {
                // whisper.cpp's whisper_lang_id falls back to english on miss.
                self.language_token("en").unwrap_or(self.sot + 1)
            });
            seq.push(lang_tok);
            seq.push(if translate {
                self.translate
            } else {
                self.transcribe
            });
        }
        if !timestamps {
            seq.push(self.no_timestamps);
        }
        seq
    }

    /// Decode token ids to text by concatenating their raw bytes and applying a
    /// single lossy UTF-8 conversion at the end. Special tokens (`>= eot`)
    /// contribute no text. This matches whisper.cpp, where text tokens hold raw
    /// byte-level BPE pieces that may split a UTF-8 character across a boundary.
    #[must_use]
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if self.is_special(id) {
                continue;
            }
            if let Some(b) = self.token_bytes(id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Like [`Tokenizer::decode`] but renders special tokens as the bracketed
    /// debug forms whisper.cpp stores in `id_to_token` (`whisper.cpp` lines
    /// 1644-1668): `[_SOT_]`, `[_EOT_]`, `[_TRANSLATE_]`, `[_TRANSCRIBE_]`,
    /// `[_SOLM_]`, `[_PREV_]`, `[_NOSP_]`, `[_NOT_]`, `[_BEG_]`, `[_TT_n]` for
    /// timestamps, and `[_LANG_xx]` for language tokens.
    #[must_use]
    pub fn decode_with_special(&self, ids: &[i32]) -> String {
        let mut out = String::new();
        let mut text_buf: Vec<u8> = Vec::new();
        let flush = |buf: &mut Vec<u8>, out: &mut String| {
            if !buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(buf));
                buf.clear();
            }
        };
        for &id in ids {
            if self.is_special(id) {
                flush(&mut text_buf, &mut out);
                out.push_str(&self.special_label(id));
            } else if let Some(b) = self.token_bytes(id) {
                text_buf.extend_from_slice(b);
            }
        }
        flush(&mut text_buf, &mut out);
        out
    }

    /// Bracketed debug label for a special token id. Order mirrors
    /// whisper.cpp's synthesis loop (lines 1644-1668): the `[_TT_n]` branch is
    /// checked first (`i > token_beg`), so `token_beg` itself renders `[_BEG_]`.
    fn special_label(&self, id: i32) -> String {
        if id > self.timestamp_begin {
            return format!("[_TT_{}]", id - self.timestamp_begin);
        }
        if id == self.eot {
            return "[_EOT_]".to_string();
        }
        if id == self.sot {
            return "[_SOT_]".to_string();
        }
        if id == self.translate {
            return "[_TRANSLATE_]".to_string();
        }
        if id == self.transcribe {
            return "[_TRANSCRIBE_]".to_string();
        }
        if id == self.solm {
            return "[_SOLM_]".to_string();
        }
        if id == self.sot_prev {
            return "[_PREV_]".to_string();
        }
        if id == self.no_speech {
            return "[_NOSP_]".to_string();
        }
        if id == self.no_timestamps {
            return "[_NOT_]".to_string();
        }
        if id == self.timestamp_begin {
            return "[_BEG_]".to_string();
        }
        // Language tokens: sot < id <= sot + num_languages (line 1664).
        if id > self.sot && id <= self.sot + self.n_langs {
            let lang_id = id - self.sot - 1;
            if let Some((code, _, _)) = LANGUAGES.iter().find(|(_, lid, _)| *lid == lang_id) {
                return format!("[_LANG_{code}]");
            }
        }
        format!("[_extra_token_{id}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn hp(n_vocab: i32, n_audio_layer: i32) -> WhisperHParams {
        WhisperHParams {
            n_vocab,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1,
        }
    }

    /// Build a tiny synthetic vocab that exercises byte concatenation and a
    /// multi-byte UTF-8 character split across two tokens.
    fn synthetic_vocab(n_vocab: i32) -> Vec<Vec<u8>> {
        // "😀" is U+1F600 = bytes F0 9F 98 80. Split across tokens 3 and 4.
        let emoji = [0xF0u8, 0x9F, 0x98, 0x80];
        let mut v: Vec<Vec<u8>> = vec![
            b"Hel".to_vec(),     // 0
            b"lo".to_vec(),      // 1
            b" world".to_vec(),  // 2
            emoji[..2].to_vec(), // 3: first half of emoji
            emoji[2..].to_vec(), // 4: second half of emoji
            vec![0xFFu8],        // 5: orphan invalid byte
            b"!".to_vec(),       // 6
        ];
        // Pad with placeholder byte tokens up to n_vocab so ids are in range.
        while (v.len() as i32) < n_vocab {
            v.push(vec![b'.']);
        }
        v
    }

    #[test]
    fn decode_joins_bytes_and_handles_split_utf8() {
        let tk = Tokenizer::from_vocab(&hp(51864, 4), synthetic_vocab(51864));
        // Tokens 3+4 reassemble the emoji that was split mid-character.
        assert_eq!(tk.decode(&[0, 1, 2, 3, 4, 6]), "Hello world😀!");
    }

    #[test]
    fn decode_lossy_replaces_orphan_byte() {
        let tk = Tokenizer::from_vocab(&hp(51864, 4), synthetic_vocab(51864));
        // Token 5 is a lone 0xFF, which is invalid UTF-8 -> U+FFFD.
        let out = tk.decode(&[0, 1, 5]);
        assert_eq!(out, "Hello\u{FFFD}");
    }

    #[test]
    fn decode_skips_special_tokens() {
        let tk = Tokenizer::from_vocab(&hp(51864, 4), synthetic_vocab(51864));
        // eot/sot contribute nothing to decoded text.
        assert_eq!(tk.decode(&[tk.sot, 0, 1, tk.eot]), "Hello");
    }

    // ----- Special-id derivation, both model families -----
    //
    // English-only (n_vocab = 51864): NOT multilingual (< 51865), so no shift
    // is applied; ids equal whisper.cpp's struct defaults (whisper.cpp 439-449).
    #[test]
    fn special_ids_english_only_51864() {
        let tk = Tokenizer::from_vocab(&hp(51864, 4), synthetic_vocab(51864));
        assert_eq!(tk.eot, 50256, "whisper.cpp:439 token_eot base");
        assert_eq!(tk.sot, 50257, "whisper.cpp:440 token_sot base");
        assert_eq!(tk.translate, 50357, "whisper.cpp:442");
        assert_eq!(tk.transcribe, 50358, "whisper.cpp:443");
        assert_eq!(tk.solm, 50359, "whisper.cpp:445");
        assert_eq!(tk.sot_prev, 50360, "whisper.cpp:446 token_prev");
        assert_eq!(tk.no_speech, 50361, "whisper.cpp:447 token_nosp");
        assert_eq!(tk.no_timestamps, 50362, "whisper.cpp:448 token_not");
        assert_eq!(tk.timestamp_begin, 50363, "whisper.cpp:449 token_beg");
        assert!(!tk.is_multilingual());
    }

    // Multilingual v3 (n_vocab = 51866): multilingual (>= 51865).
    //   num_languages = 51866 - 51765 - 1 = 100   (whisper.cpp:456)
    //   dt = num_languages - 98 = 2               (whisper.cpp:1630)
    //   eot = 50256 + 1 = 50257                   (whisper.cpp:1626)
    //   sot = 50257 + 1 = 50258                   (whisper.cpp:1627)
    //   translate  = 50357 + 2 = 50359            (whisper.cpp:1632)
    //   transcribe = 50358 + 2 = 50360            (whisper.cpp:1633)
    //   solm       = 50359 + 2 = 50361            (whisper.cpp:1634)
    //   prev       = 50360 + 2 = 50362            (whisper.cpp:1635)
    //   nosp       = 50361 + 2 = 50363            (whisper.cpp:1636)
    //   not        = 50362 + 2 = 50364            (whisper.cpp:1637)
    //   beg        = 50363 + 2 = 50365            (whisper.cpp:1638)
    //   lang base  = sot + 1 = 50259              (whisper.cpp:4248)
    #[test]
    fn special_ids_multilingual_v3_51866() {
        let tk = Tokenizer::from_vocab(&hp(51866, 32), synthetic_vocab(51866));
        assert_eq!(tk.num_languages(), 100);
        assert_eq!(tk.eot, 50257);
        assert_eq!(tk.sot, 50258);
        assert_eq!(tk.translate, 50359);
        assert_eq!(tk.transcribe, 50360);
        assert_eq!(tk.solm, 50361);
        assert_eq!(tk.sot_prev, 50362);
        assert_eq!(tk.no_speech, 50363);
        assert_eq!(tk.no_timestamps, 50364);
        assert_eq!(tk.timestamp_begin, 50365);
        // Language layout: en is sot+1, the base of the language block.
        assert_eq!(tk.language_token("en"), Some(50259));
        assert_eq!(tk.language_token("zh"), Some(50260));
        assert_eq!(tk.language_token("yue"), Some(50259 + 99));
        assert!(tk.is_multilingual());
    }

    #[test]
    fn timestamp_helpers() {
        let tk = Tokenizer::from_vocab(&hp(51866, 32), synthetic_vocab(51866));
        assert!(!tk.is_timestamp(tk.timestamp_begin - 1));
        assert!(tk.is_timestamp(tk.timestamp_begin));
        assert!((tk.timestamp_sec(tk.timestamp_begin) - 0.0).abs() < 1e-6);
        assert!((tk.timestamp_sec(tk.timestamp_begin + 100) - 2.0).abs() < 1e-6);
        assert!((tk.timestamp_sec(tk.timestamp_begin + 1) - 0.02).abs() < 1e-6);
    }

    #[test]
    fn sot_sequence_english_only_has_no_lang_or_task() {
        let tk = Tokenizer::from_vocab(&hp(51864, 4), synthetic_vocab(51864));
        // No language/task tokens for English-only models.
        assert_eq!(tk.sot_sequence(None, false, true), vec![tk.sot]);
        assert_eq!(
            tk.sot_sequence(None, false, false),
            vec![tk.sot, tk.no_timestamps]
        );
        // language/translate flags are ignored for english-only.
        assert_eq!(tk.sot_sequence(Some("fr"), true, true), vec![tk.sot]);
        assert_eq!(tk.language_token("en"), None);
    }

    #[test]
    fn sot_sequence_multilingual_layout() {
        let tk = Tokenizer::from_vocab(&hp(51866, 32), synthetic_vocab(51866));
        let en = tk.language_token("en").unwrap();
        assert_eq!(
            tk.sot_sequence(Some("en"), false, true),
            vec![tk.sot, en, tk.transcribe]
        );
        assert_eq!(
            tk.sot_sequence(Some("en"), true, false),
            vec![tk.sot, en, tk.translate, tk.no_timestamps]
        );
        // None defaults to english.
        assert_eq!(
            tk.sot_sequence(None, false, true),
            vec![tk.sot, en, tk.transcribe]
        );
        // Unknown language falls back to english (whisper_lang_id default).
        assert_eq!(
            tk.sot_sequence(Some("xx"), false, true),
            vec![tk.sot, en, tk.transcribe]
        );
    }

    #[test]
    fn decode_with_special_renders_bracketed_forms() {
        let tk = Tokenizer::from_vocab(&hp(51866, 32), synthetic_vocab(51866));
        let s = tk.decode_with_special(&[
            tk.sot,
            tk.language_token("en").unwrap(),
            tk.transcribe,
            0,
            1,
            tk.timestamp_begin,
            tk.timestamp_begin + 50,
            tk.eot,
        ]);
        assert_eq!(
            s,
            "[_SOT_][_LANG_en][_TRANSCRIBE_]Hello[_BEG_][_TT_50][_EOT_]"
        );
    }

    #[test]
    fn non_speech_set_matches_symbols() {
        // Build a vocab where a couple of non-speech symbols are present.
        let mut v = synthetic_vocab(51864);
        v[10] = b"\"".to_vec(); // bare quote
        v[11] = b" (".to_vec(); // space-prefixed paren
        v[12] = b" -".to_vec(); // special hyphen case
        v[13] = b"\"".to_vec(); // duplicate: first matching id wins
        let tk = Tokenizer::from_vocab(&hp(51864, 4), v);
        let ns = tk.non_speech_tokens();
        assert!(ns.contains(&10), "bare quote suppressed");
        assert!(ns.contains(&11), "space-prefixed paren suppressed");
        assert!(ns.contains(&12), "space-hyphen suppressed");
        assert!(!ns.contains(&13), "duplicate token id is not added");
        // Sorted and de-duplicated.
        let mut sorted = ns.to_vec();
        sorted.sort_unstable();
        assert_eq!(ns, sorted.as_slice());
    }

    // ----- Gated test against a real tiny.en model -----

    /// Minimal self-contained ggml reader: magic + 11 hparams + filterbank
    /// (skipped) + vocab. Independent of the `ggml` module being written in
    /// parallel. Returns `(hparams, vocab_tokens)`.
    fn read_tiny_en_vocab(path: &std::path::Path) -> Option<(WhisperHParams, Vec<Vec<u8>>)> {
        let mut f = std::fs::File::open(path).ok()?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).ok()?;
        let mut pos = 0usize;
        let read_u32 = |pos: &mut usize| -> Option<u32> {
            let b = buf.get(*pos..*pos + 4)?;
            *pos += 4;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let magic = read_u32(&mut pos)?;
        assert_eq!(magic, 0x6767_6d6c, "ggml magic");
        let n_vocab = read_u32(&mut pos)? as i32;
        let n_audio_ctx = read_u32(&mut pos)? as i32;
        let n_audio_state = read_u32(&mut pos)? as i32;
        let n_audio_head = read_u32(&mut pos)? as i32;
        let n_audio_layer = read_u32(&mut pos)? as i32;
        let n_text_ctx = read_u32(&mut pos)? as i32;
        let n_text_state = read_u32(&mut pos)? as i32;
        let n_text_head = read_u32(&mut pos)? as i32;
        let n_text_layer = read_u32(&mut pos)? as i32;
        let n_mels = read_u32(&mut pos)? as i32;
        let ftype = read_u32(&mut pos)? as i32;
        // mel filters: n_mel, n_fft, then n_mel*n_fft f32 values (skipped).
        let f_n_mel = read_u32(&mut pos)? as usize;
        let f_n_fft = read_u32(&mut pos)? as usize;
        pos += f_n_mel * f_n_fft * 4;
        // vocab
        let vsize = read_u32(&mut pos)? as usize;
        let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(vsize);
        for _ in 0..vsize {
            let len = read_u32(&mut pos)? as usize;
            let bytes = buf.get(pos..pos + len)?.to_vec();
            pos += len;
            tokens.push(bytes);
        }
        let hparams = WhisperHParams {
            n_vocab,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels,
            ftype,
        };
        Some((hparams, tokens))
    }

    #[test]
    fn gated_tiny_en_vocab() {
        let Some(path) = super::super::find_model_file("tiny.en") else {
            eprintln!("skipping gated_tiny_en_vocab: tiny.en model not found");
            return;
        };
        let Some((hparams, tokens)) = read_tiny_en_vocab(&path) else {
            panic!("failed to read tiny.en vocab from {}", path.display());
        };
        let tk = Tokenizer::from_vocab(&hparams, tokens);

        // English-only model.
        assert!(!tk.is_multilingual(), "tiny.en is English-only");
        assert_eq!(tk.eot, 50256);
        assert_eq!(tk.sot, 50257);
        assert_eq!(tk.timestamp_begin, 50363);

        // A common token " the" exists in the GPT-2 byte-level vocab.
        let has_the = (0..tk.vocab_size()).any(|id| tk.token_bytes(id) == Some(b" the".as_slice()));
        assert!(has_the, "vocab should contain a ' the' token");

        // Timestamp arithmetic on the real model.
        assert!((tk.timestamp_sec(tk.timestamp_begin) - 0.0).abs() < 1e-6);
        assert!((tk.timestamp_sec(tk.timestamp_begin + 100) - 2.0).abs() < 1e-6);

        // English-only sot sequence is just [sot] (no lang/task tokens).
        assert_eq!(tk.sot_sequence(None, false, true), vec![tk.sot]);

        // language_token returns None for English-only models.
        assert_eq!(tk.language_token("en"), None);
    }
}
