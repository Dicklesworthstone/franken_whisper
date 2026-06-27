//! Model-weights infrastructure for **non-Whisper** neural models.
//!
//! Epic B (neural diarization / source separation) needs to load auxiliary
//! model weights — first up, the ECAPA-TDNN speaker-embedding network consumed
//! by [`bd-ohex`](.beads/) — that are *not* whisper.cpp ggml files. The
//! community-standard interchange format for such weights is **safetensors**,
//! so this module provides:
//!
//! - [`SafetensorsFile`] — a strict, self-contained safetensors reader that
//!   loads any `F32` / `F16` / `BF16` tensor and exposes it as `(shape, Vec<f32>)`.
//! - [`WeightsManifest`] + [`validate`] — an expected-census check so a wrong
//!   or stale file fails *loud* with a named diff (missing / mis-shaped / extra
//!   tensors) instead of silently mis-loading.
//! - [`resolve_aux_model`] — locate an aux model file across the same search
//!   dirs the whisper engine uses, plus an `aux/` subdirectory of each.
//!
//! # Why a local safetensors reader (and not `ft-serialize`)
//!
//! FrankenTorch's `ft-serialize` crate already ships a battle-tested
//! [`load_safetensors`](https://docs.rs) (`crates/ft-serialize/src/lib.rs`,
//! `load_safetensors` / `load_safetensors_from_bytes`, with `F16`/`BF16`→`f32`
//! conversion). Adding it as a dependency, however, requires editing
//! `Cargo.toml`, which is out of scope for the bead that introduced this module
//! (bd-r95i). The safetensors container is trivial — a `u64` little-endian
//! header length, a JSON header mapping tensor name → `{dtype, shape,
//! data_offsets:[begin,end]}` (plus an optional `__metadata__` object), then
//! the raw little-endian tensor bytes — so we implement a ~native reader here
//! (≈150 lines) and keep the dependency tree unchanged. The float conversions
//! reuse `ft_core`'s `half`-backed `Float16` / `BFloat16` (already a dep, same
//! types `ggml.rs` uses), so there is no numerical divergence from the rest of
//! the engine. **Revisit** unifying on `ft-serialize::load_safetensors` in
//! bd-2th6 / bd-to6k once a Cargo edge is permissible.
//!
//! # No network access
//!
//! Like the whisper engine, nothing here ever downloads. Fetching / converting
//! aux models is an explicit, separate, user-invoked step (see
//! `scripts/fetch_aux_models.sh` and `scripts/convert_to_safetensors.py`); a
//! missing model is a hard, actionable error from [`resolve_aux_model`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ft_core::{BFloat16, Float16};
use serde_json::Value;

use crate::error::{FwError, FwResult};

/// Hard ceiling on the JSON header length (100 MB). A safetensors header is at
/// most a few MB even for very large models; anything larger is corruption or
/// a hostile file and is rejected before we attempt to allocate / parse it.
const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

/// The reserved key inside the JSON header that carries free-form, non-tensor
/// metadata (e.g. provenance / conversion info written by
/// `convert_to_safetensors.py`).
const METADATA_KEY: &str = "__metadata__";

/// A safetensors dtype we know how to materialize as `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StDType {
    F32,
    F16,
    Bf16,
}

impl StDType {
    /// Parse the dtype string from a header entry. Only the dense float types
    /// the neural aux models use are supported; integer / bool / f64 dtypes are
    /// rejected loudly (the consumers want f32 weights).
    fn parse(s: &str, tensor: &str) -> FwResult<Self> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            other => Err(FwError::InvalidRequest(format!(
                "safetensors tensor `{tensor}`: unsupported dtype `{other}` \
                 (only F32, F16, BF16 are supported)"
            ))),
        }
    }

    /// Bytes per element of the on-disk representation.
    const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

/// One parsed tensor directory entry: its dtype, logical shape, and the
/// `[begin, end)` byte span (relative to the start of the data section) holding
/// its little-endian payload.
#[derive(Debug, Clone)]
struct TensorEntry {
    dtype: StDType,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// A loaded safetensors file: the parsed tensor directory plus the raw data
/// section bytes, ready to materialize any tensor as `f32` on demand.
///
/// Tensors are stored sorted by name (a [`BTreeMap`]) so [`names`](Self::names)
/// yields a stable, sorted order. Construct via [`load`](Self::load).
#[derive(Debug)]
pub struct SafetensorsFile {
    /// Tensor directory, keyed by name (sorted).
    tensors: BTreeMap<String, TensorEntry>,
    /// Optional `__metadata__` object from the header, preserved verbatim.
    metadata: Option<Value>,
    /// The raw tensor data section (everything after the JSON header).
    data: Vec<u8>,
}

impl SafetensorsFile {
    /// Load and strictly validate the safetensors file at `path`.
    ///
    /// Validation performed (each failure is an [`FwError::InvalidRequest`]
    /// naming the offending tensor / field):
    /// - the 8-byte header length is present and `< 100 MB`;
    /// - the JSON header parses and every non-metadata value is an object with
    ///   a string `dtype`, an array `shape` of non-negative integers, and a
    ///   2-element `data_offsets` `[begin, end]` with `begin <= end`;
    /// - every tensor's `[begin, end)` lies within the data section;
    /// - `shape.product() * dtype.byte_width() == end - begin` (the declared
    ///   shape exactly accounts for the byte span).
    ///
    /// Overlapping spans are *not* rejected (the format permits aliasing); only
    /// in-bounds and exact-width are required.
    ///
    /// # Duplicate tensor names
    ///
    /// The safetensors spec disallows duplicate keys in the JSON header, but we
    /// do not reject them: the header is parsed with [`serde_json`], whose
    /// object [`Map`](serde_json::Map) keeps the **last** value for a repeated
    /// key. A header containing the same tensor name twice therefore resolves
    /// *last-wins* — the final occurrence's `dtype`/`shape`/`data_offsets` are
    /// the ones validated and exposed; earlier duplicates are silently dropped
    /// before this function ever sees them.
    ///
    /// # Errors
    ///
    /// [`FwError::Io`] on read failure; [`FwError::Json`] on a malformed header
    /// JSON; [`FwError::InvalidRequest`] for any structural violation above.
    pub fn load(path: &Path) -> FwResult<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse from an in-memory safetensors byte buffer (the testable core of
    /// [`load`](Self::load)).
    ///
    /// Duplicate tensor names in the JSON header resolve *last-wins* via
    /// [`serde_json::Map`] semantics — the spec disallows duplicate keys, but we
    /// accept them and keep the final occurrence. See [`load`](Self::load) for
    /// the full contract.
    ///
    /// # Errors
    ///
    /// See [`load`](Self::load).
    pub fn from_bytes(bytes: &[u8]) -> FwResult<Self> {
        if bytes.len() < 8 {
            return Err(FwError::InvalidRequest(format!(
                "safetensors file too short: {} bytes (need at least an 8-byte header length)",
                bytes.len()
            )));
        }
        let header_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        if header_len > MAX_HEADER_LEN {
            return Err(FwError::InvalidRequest(format!(
                "safetensors header length {header_len} exceeds the {MAX_HEADER_LEN}-byte sanity cap"
            )));
        }
        let header_len = usize::try_from(header_len).map_err(|_| {
            FwError::InvalidRequest(format!(
                "safetensors header length {header_len} does not fit in usize on this platform"
            ))
        })?;
        let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
            FwError::InvalidRequest("safetensors header length overflows file offset".to_owned())
        })?;
        if header_end > bytes.len() {
            return Err(FwError::InvalidRequest(format!(
                "safetensors header claims {header_len} bytes but only {} remain after the length prefix",
                bytes.len().saturating_sub(8)
            )));
        }

        let header_bytes = &bytes[8..header_end];
        let header: Value = serde_json::from_slice(header_bytes)?;
        let obj = header.as_object().ok_or_else(|| {
            FwError::InvalidRequest("safetensors header is not a JSON object".to_owned())
        })?;

        let data = bytes[header_end..].to_vec();
        let data_len = data.len();

        let mut metadata = None;
        let mut tensors = BTreeMap::new();
        for (name, value) in obj {
            if name == METADATA_KEY {
                metadata = Some(value.clone());
                continue;
            }
            let entry = parse_tensor_entry(name, value, data_len)?;
            tensors.insert(name.clone(), entry);
        }

        Ok(Self {
            tensors,
            metadata,
            data,
        })
    }

    /// Materialize tensor `name` as `(shape, row-major f32 data)`.
    ///
    /// `F16` and `BF16` payloads are converted to `f32` via `ft_core`'s
    /// `half`-backed types (identical to the whisper ggml path), so the values
    /// match the rest of the engine bit-for-bit.
    ///
    /// # Errors
    ///
    /// [`FwError::InvalidRequest`] if `name` is absent (the message lists a few
    /// available names), or — defensively — if the recorded byte span no longer
    /// matches the element count (should be impossible post-[`load`](Self::load),
    /// but checked rather than panicking on a slice).
    pub fn tensor_f32(&self, name: &str) -> FwResult<(Vec<usize>, Vec<f32>)> {
        let entry = self.tensors.get(name).ok_or_else(|| {
            let mut available: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
            available.sort_unstable();
            let preview: Vec<&str> = available.iter().take(8).copied().collect();
            let suffix = if available.len() > preview.len() {
                format!(", … ({} total)", available.len())
            } else {
                String::new()
            };
            FwError::InvalidRequest(format!(
                "safetensors tensor `{name}` not found; available: [{}]{suffix}",
                preview.join(", ")
            ))
        })?;

        let raw = &self.data[entry.begin..entry.end];
        let n_elements: usize = entry.shape.iter().product();
        let width = entry.dtype.byte_width();
        if raw.len() != n_elements * width {
            return Err(FwError::InvalidRequest(format!(
                "safetensors tensor `{name}`: byte span {} != {} elements * {} bytes",
                raw.len(),
                n_elements,
                width
            )));
        }

        let mut out = Vec::with_capacity(n_elements);
        match entry.dtype {
            StDType::F32 => {
                for chunk in raw.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
            }
            StDType::F16 => {
                for chunk in raw.as_chunks::<2>().0 {
                    out.push(Float16::from_le_bytes(*chunk).to_f32());
                }
            }
            StDType::Bf16 => {
                for chunk in raw.as_chunks::<2>().0 {
                    out.push(BFloat16::from_le_bytes(*chunk).to_f32());
                }
            }
        }
        Ok((entry.shape.clone(), out))
    }

    /// The shape of tensor `name`, without materializing its data.
    ///
    /// # Errors
    ///
    /// [`FwError::InvalidRequest`] if `name` is absent.
    pub fn shape(&self, name: &str) -> FwResult<&[usize]> {
        self.tensors
            .get(name)
            .map(|e| e.shape.as_slice())
            .ok_or_else(|| {
                FwError::InvalidRequest(format!("safetensors tensor `{name}` not found"))
            })
    }

    /// Iterate tensor names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The number of tensors in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file has no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The `__metadata__` object from the header, if present.
    #[must_use]
    pub fn metadata(&self) -> Option<&Value> {
        self.metadata.as_ref()
    }
}

/// Parse and validate one tensor directory entry against the data section length.
fn parse_tensor_entry(name: &str, value: &Value, data_len: usize) -> FwResult<TensorEntry> {
    let obj = value.as_object().ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: entry is not a JSON object"
        ))
    })?;

    let dtype_str = obj.get("dtype").and_then(Value::as_str).ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: missing or non-string `dtype`"
        ))
    })?;
    let dtype = StDType::parse(dtype_str, name)?;

    let shape_arr = obj.get("shape").and_then(Value::as_array).ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: missing or non-array `shape`"
        ))
    })?;
    let mut shape = Vec::with_capacity(shape_arr.len());
    for dim in shape_arr {
        let d = dim.as_u64().ok_or_else(|| {
            FwError::InvalidRequest(format!(
                "safetensors tensor `{name}`: shape dimension `{dim}` is not a non-negative integer"
            ))
        })?;
        shape.push(usize::try_from(d).map_err(|_| {
            FwError::InvalidRequest(format!(
                "safetensors tensor `{name}`: shape dimension {d} does not fit in usize"
            ))
        })?);
    }

    let offsets = obj
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            FwError::InvalidRequest(format!(
                "safetensors tensor `{name}`: missing or non-array `data_offsets`"
            ))
        })?;
    if offsets.len() != 2 {
        return Err(FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: `data_offsets` must have exactly 2 elements, got {}",
            offsets.len()
        )));
    }
    let begin = offset_value(name, &offsets[0], "begin")?;
    let end = offset_value(name, &offsets[1], "end")?;

    if begin > end {
        return Err(FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: data_offsets begin {begin} > end {end}"
        )));
    }
    if end > data_len {
        return Err(FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: data_offsets end {end} exceeds data section length {data_len}"
        )));
    }

    let span = end - begin;
    let n_elements: usize = shape.iter().product();
    let expected = n_elements.checked_mul(dtype.byte_width()).ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: shape element count overflows when sized"
        ))
    })?;
    if span != expected {
        return Err(FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: byte span {span} != shape product {n_elements} \
             * {} bytes ({expected}) for dtype {dtype_str}",
            dtype.byte_width()
        )));
    }

    Ok(TensorEntry {
        dtype,
        shape,
        begin,
        end,
    })
}

/// Parse one `data_offsets` element as a `usize` byte offset.
fn offset_value(name: &str, value: &Value, which: &str) -> FwResult<usize> {
    let n = value.as_u64().ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: data_offsets {which} `{value}` is not a non-negative integer"
        ))
    })?;
    usize::try_from(n).map_err(|_| {
        FwError::InvalidRequest(format!(
            "safetensors tensor `{name}`: data_offsets {which} {n} does not fit in usize"
        ))
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Expected-census manifest
// ─────────────────────────────────────────────────────────────────────────

/// An expected tensor census (name → shape) for a particular model
/// architecture.
///
/// Epic B model loaders declare a `WeightsManifest` describing exactly which
/// tensors they require and at what shapes; [`validate`] then checks a loaded
/// [`SafetensorsFile`] against it and fails *loud* (a named diff) on any
/// mismatch, so a wrong / stale / truncated weights file is caught at load time
/// rather than producing silently-garbage embeddings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightsManifest {
    /// `(tensor name, expected logical shape)` pairs.
    pub entries: Vec<(String, Vec<usize>)>,
}

impl WeightsManifest {
    /// Build a manifest from any iterator of `(name, shape)` pairs.
    pub fn new<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = (S, Vec<usize>)>,
        S: Into<String>,
    {
        Self {
            entries: entries
                .into_iter()
                .map(|(name, shape)| (name.into(), shape))
                .collect(),
        }
    }

    /// The number of declared tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the manifest declares no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Validate a loaded [`SafetensorsFile`] against an expected [`WeightsManifest`].
///
/// On success every manifest tensor is present at exactly its declared shape.
/// On failure the returned [`FwError::InvalidRequest`] carries a **loud diff**
/// listing, in order:
/// - **missing** tensors (declared but absent from the file),
/// - **shape mismatches** (`name: expected [..] got [..]`),
/// - **extra** tensors (present in the file but not declared).
///
/// Extras are reported but, on their own, are *not* fatal-by-omission: a file
/// may legitimately carry buffers a given loader ignores. They are surfaced so
/// an operator can see the full picture; the call still fails whenever there is
/// any missing tensor or shape mismatch, and also when there are only extras
/// (a census mismatch the caller asked us to enforce).
///
/// # Errors
///
/// [`FwError::InvalidRequest`] with the diff report when the census does not
/// match exactly.
pub fn validate(file: &SafetensorsFile, manifest: &WeightsManifest) -> FwResult<()> {
    use std::collections::BTreeSet;
    use std::fmt::Write as _;

    let mut missing: Vec<&str> = Vec::new();
    let mut mismatched: Vec<(String, Vec<usize>, Vec<usize>)> = Vec::new();
    let declared: BTreeSet<&str> = manifest.entries.iter().map(|(n, _)| n.as_str()).collect();

    for (name, expected) in &manifest.entries {
        match file.tensors.get(name) {
            None => missing.push(name.as_str()),
            Some(entry) if &entry.shape != expected => {
                mismatched.push((name.clone(), expected.clone(), entry.shape.clone()));
            }
            Some(_) => {}
        }
    }

    let extras: Vec<&str> = file
        .tensors
        .keys()
        .map(String::as_str)
        .filter(|n| !declared.contains(n))
        .collect();

    if missing.is_empty() && mismatched.is_empty() && extras.is_empty() {
        return Ok(());
    }

    let mut report = String::from(
        "weights manifest validation FAILED: loaded safetensors file does not match the \
         expected tensor census.",
    );
    if !missing.is_empty() {
        let _ = write!(report, "\n  MISSING ({}):", missing.len());
        for name in &missing {
            let _ = write!(report, "\n    - {name}");
        }
    }
    if !mismatched.is_empty() {
        let _ = write!(report, "\n  SHAPE MISMATCH ({}):", mismatched.len());
        for (name, expected, got) in &mismatched {
            let _ = write!(report, "\n    - {name}: expected {expected:?} got {got:?}");
        }
    }
    if !extras.is_empty() {
        let _ = write!(report, "\n  EXTRA ({}):", extras.len());
        for name in &extras {
            let _ = write!(report, "\n    - {name}");
        }
    }
    Err(FwError::InvalidRequest(report))
}

// ─────────────────────────────────────────────────────────────────────────
// Aux-model resolution
// ─────────────────────────────────────────────────────────────────────────

/// The base directories searched for auxiliary (non-whisper) model files.
///
/// This **duplicates** the precedence list in
/// [`super::model_search_dirs`](super) because that function is private and the
/// bead introducing this module is not permitted to change its visibility.
/// **Unify** the two into a single shared accessor in bd-0522.
///
/// Precedence (highest first), mirroring the whisper engine:
/// 1. `$FRANKEN_WHISPER_MODEL_DIR`
/// 2. `$FRANKEN_WHISPER_TEST_MODEL_DIR`
/// 3. `~/.cache/franken_whisper/models`
/// 4. `~/.cache/franken_whisper/test-models`
/// 5. `~/models/whisper`
fn aux_base_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in [
        "FRANKEN_WHISPER_MODEL_DIR",
        "FRANKEN_WHISPER_TEST_MODEL_DIR",
    ] {
        if let Ok(dir) = std::env::var(var)
            && !dir.is_empty()
        {
            dirs.push(PathBuf::from(dir));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".cache").join("franken_whisper").join("models"));
        dirs.push(
            home.join(".cache")
                .join("franken_whisper")
                .join("test-models"),
        );
        dirs.push(home.join("models").join("whisper"));
    }
    dirs
}

/// Expand the base dirs into the full ordered aux search list: for each base
/// dir, the dir itself **and** its `aux/` subdirectory. The aux subdir is
/// listed first within each base so a curated `aux/` copy wins over a stray
/// top-level file.
fn aux_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for base in aux_base_dirs() {
        dirs.push(base.join("aux"));
        dirs.push(base);
    }
    dirs
}

/// Resolve an auxiliary model `file_name` (e.g. `"ecapa_tdnn_voxceleb.safetensors"`)
/// to a concrete path.
///
/// Two forms are accepted, mirroring [`super::resolve_model`]:
/// 1. an existing filesystem path (returned canonicalized);
/// 2. a bare file name, searched across [`aux_search_dirs`] (each whisper search
///    dir plus its `aux/` subdirectory), first match wins.
///
/// # No network access
///
/// Never downloads. A miss is a hard, actionable error pointing the operator at
/// `scripts/fetch_aux_models.sh`.
///
/// # Errors
///
/// [`FwError::InvalidRequest`] when nothing resolves (the message enumerates
/// every searched location); [`FwError::Io`] if canonicalizing an existing path
/// fails.
pub fn resolve_aux_model(file_name: &str) -> FwResult<PathBuf> {
    let as_path = Path::new(file_name);
    if as_path.is_file() {
        return Ok(as_path.canonicalize()?);
    }
    resolve_aux_model_in_dirs(file_name, &aux_search_dirs())
}

/// Resolve a bare `file_name` against an explicit, ordered list of `dirs`
/// (first match wins). Factored out so the precedence logic is unit-testable
/// without mutating process environment variables.
fn resolve_aux_model_in_dirs(file_name: &str, dirs: &[PathBuf]) -> FwResult<PathBuf> {
    for dir in dirs {
        let candidate = dir.join(file_name);
        if candidate.is_file() {
            return Ok(candidate.canonicalize()?);
        }
    }
    Err(FwError::InvalidRequest(aux_resolution_error(
        file_name, dirs,
    )))
}

/// Build the actionable "aux model not found" message.
fn aux_resolution_error(file_name: &str, dirs: &[PathBuf]) -> String {
    use std::fmt::Write as _;
    let mut msg = format!(
        "no auxiliary model found for `{file_name}`: it is neither an existing file path \
         nor a bare name resolvable in the aux search dirs.\n\
         Searched directories (in order):"
    );
    if dirs.is_empty() {
        msg.push_str(
            "\n  (none — set $FRANKEN_WHISPER_MODEL_DIR or $HOME to enable bare-name lookup)",
        );
    } else {
        for dir in dirs {
            let _ = write!(msg, "\n  - {}", dir.join(file_name).display());
        }
    }
    msg.push_str(
        "\nFix: run `scripts/fetch_aux_models.sh` to provision aux models into the \
         `aux/` subdirectory of $FRANKEN_WHISPER_MODEL_DIR, set $FRANKEN_WHISPER_MODEL_DIR \
         to the directory containing the file, or pass an explicit path. FrankenWhisper \
         never downloads models automatically (data never leaves the machine).",
    );
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::path::Path;

    // ─────────────────────────────────────────────────────────────────────
    // Synthetic safetensors construction (no external files)
    // ─────────────────────────────────────────────────────────────────────

    /// A tensor to embed in a synthetic file: name, dtype string, shape, and
    /// the raw little-endian payload bytes.
    struct SynTensor {
        name: &'static str,
        dtype: &'static str,
        shape: Vec<usize>,
        payload: Vec<u8>,
    }

    /// Build a valid safetensors byte buffer from `tensors` (laid out
    /// contiguously in the given order) plus an optional `__metadata__` object.
    fn build_safetensors(tensors: &[SynTensor], metadata: Option<Value>) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        if let Some(meta) = metadata {
            header.insert(METADATA_KEY.to_owned(), meta);
        }
        let mut data = Vec::new();
        for t in tensors {
            let begin = data.len();
            data.extend_from_slice(&t.payload);
            let end = data.len();
            header.insert(
                t.name.to_owned(),
                serde_json::json!({
                    "dtype": t.dtype,
                    "shape": t.shape,
                    "data_offsets": [begin, end],
                }),
            );
        }
        let header_json = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_json);
        out.extend_from_slice(&data);
        out
    }

    fn f32_payload(values: &[f32]) -> Vec<u8> {
        let mut v = Vec::with_capacity(values.len() * 4);
        for &x in values {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    /// f16 bit patterns -> little-endian bytes. (0x3C00 = 1.0, 0x0000 = 0.0,
    /// 0xC000 = -2.0, 0x3800 = 0.5.)
    fn f16_payload(bits: &[u16]) -> Vec<u8> {
        let mut v = Vec::with_capacity(bits.len() * 2);
        for &b in bits {
            v.extend_from_slice(&b.to_le_bytes());
        }
        v
    }

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("fw_weights_{tag}_{pid}_{n}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_file(dir: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(contents).expect("write file");
        path
    }

    // ─────────────────────────────────────────────────────────────────────
    // SafetensorsFile load + tensor_f32 + names + metadata
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn load_two_tensors_f32_and_f16_with_metadata() {
        let metadata = serde_json::json!({ "source": "synthetic", "format": "pt" });
        let tensors = vec![
            SynTensor {
                name: "w_f32",
                dtype: "F32",
                shape: vec![2, 3],
                payload: f32_payload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            },
            SynTensor {
                name: "a_f16",
                dtype: "F16",
                shape: vec![2, 2],
                // 1.0, 0.0, -2.0, 0.5
                payload: f16_payload(&[0x3C00, 0x0000, 0xC000, 0x3800]),
            },
        ];
        let bytes = build_safetensors(&tensors, Some(metadata.clone()));
        let file = SafetensorsFile::from_bytes(&bytes).expect("load");

        // names() are sorted: "a_f16" < "w_f32".
        let names: Vec<&str> = file.names().collect();
        assert_eq!(names, vec!["a_f16", "w_f32"]);
        assert_eq!(file.len(), 2);
        assert!(!file.is_empty());

        let (shape, vals) = file.tensor_f32("w_f32").expect("decode f32");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let (shape, vals) = file.tensor_f32("a_f16").expect("decode f16");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(vals, vec![1.0, 0.0, -2.0, 0.5]);

        assert_eq!(file.metadata(), Some(&metadata));
        assert_eq!(file.shape("w_f32").expect("shape"), &[2, 3]);
    }

    #[test]
    fn bf16_tensor_decodes() {
        // bf16 = high 16 bits of f32. 1.0f32 = 0x3F800000 -> bf16 0x3F80.
        // -2.0f32 = 0xC0000000 -> bf16 0xC000.
        let payload = {
            let mut v = Vec::new();
            for &bits in &[0x3F80u16, 0xC000u16] {
                v.extend_from_slice(&bits.to_le_bytes());
            }
            v
        };
        let tensors = vec![SynTensor {
            name: "b",
            dtype: "BF16",
            shape: vec![2],
            payload,
        }];
        let bytes = build_safetensors(&tensors, None);
        let file = SafetensorsFile::from_bytes(&bytes).expect("load");
        let (shape, vals) = file.tensor_f32("b").expect("decode bf16");
        assert_eq!(shape, vec![2]);
        assert_eq!(vals, vec![1.0, -2.0]);
        // No metadata key present.
        assert!(file.metadata().is_none());
    }

    #[test]
    fn missing_tensor_error_lists_available() {
        let tensors = vec![SynTensor {
            name: "present",
            dtype: "F32",
            shape: vec![1],
            payload: f32_payload(&[7.0]),
        }];
        let bytes = build_safetensors(&tensors, None);
        let file = SafetensorsFile::from_bytes(&bytes).expect("load");
        let err = file.tensor_f32("absent").expect_err("should miss");
        let msg = err.to_string();
        assert!(msg.contains("absent"), "names missing tensor: {msg}");
        assert!(msg.contains("present"), "lists available: {msg}");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Strict validation failures
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn too_short_file_errors() {
        let err = SafetensorsFile::from_bytes(b"abc").expect_err("too short");
        assert!(matches!(err, FwError::InvalidRequest(_)));
    }

    #[test]
    fn header_len_beyond_file_errors() {
        // Claim a huge header but provide no header bytes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(1_000u64).to_le_bytes());
        bytes.extend_from_slice(b"{}"); // only 2 header bytes present
        let err = SafetensorsFile::from_bytes(&bytes).expect_err("header overruns");
        let msg = err.to_string();
        assert!(msg.contains("header"), "msg: {msg}");
    }

    #[test]
    fn header_len_over_cap_errors() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(MAX_HEADER_LEN + 1).to_le_bytes());
        let err = SafetensorsFile::from_bytes(&bytes).expect_err("over cap");
        let msg = err.to_string();
        assert!(msg.contains("sanity cap"), "msg: {msg}");
    }

    #[test]
    fn offsets_out_of_bounds_errors() {
        // Hand-craft a header whose data_offsets exceed the (empty) data
        // section, bypassing the safe builder.
        let header = serde_json::json!({
            "t": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] }
        });
        let header_json = serde_json::to_vec(&header).expect("ser");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_json);
        // No data section bytes at all -> end (4) > data_len (0).
        let err = SafetensorsFile::from_bytes(&bytes).expect_err("oob");
        let msg = err.to_string();
        assert!(msg.contains("exceeds data section length"), "msg: {msg}");
        assert!(msg.contains('t'), "names tensor: {msg}");
    }

    #[test]
    fn shape_byte_span_mismatch_errors() {
        // shape product (3) * 4 bytes = 12, but provide an 8-byte span.
        let header = serde_json::json!({
            "t": { "dtype": "F32", "shape": [3], "data_offsets": [0, 8] }
        });
        let header_json = serde_json::to_vec(&header).expect("ser");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_json);
        bytes.extend_from_slice(&[0u8; 8]);
        let err = SafetensorsFile::from_bytes(&bytes).expect_err("span mismatch");
        let msg = err.to_string();
        assert!(msg.contains("byte span"), "msg: {msg}");
    }

    #[test]
    fn unsupported_dtype_errors() {
        let header = serde_json::json!({
            "t": { "dtype": "I64", "shape": [1], "data_offsets": [0, 8] }
        });
        let header_json = serde_json::to_vec(&header).expect("ser");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_json);
        bytes.extend_from_slice(&[0u8; 8]);
        let err = SafetensorsFile::from_bytes(&bytes).expect_err("bad dtype");
        let msg = err.to_string();
        assert!(msg.contains("unsupported dtype"), "msg: {msg}");
        assert!(msg.contains("I64"), "names dtype: {msg}");
    }

    #[test]
    fn load_from_disk_round_trips() {
        let dir = TempDir::new("disk");
        let tensors = vec![SynTensor {
            name: "x",
            dtype: "F32",
            shape: vec![2],
            payload: f32_payload(&[3.5, -1.25]),
        }];
        let bytes = build_safetensors(&tensors, None);
        let path = write_file(dir.path(), "m.safetensors", &bytes);
        let file = SafetensorsFile::load(&path).expect("load from disk");
        let (_, vals) = file.tensor_f32("x").expect("decode");
        assert_eq!(vals, vec![3.5, -1.25]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // WeightsManifest::validate happy + loud-diff
    // ─────────────────────────────────────────────────────────────────────

    fn census_file() -> SafetensorsFile {
        let tensors = vec![
            SynTensor {
                name: "layer.weight",
                dtype: "F32",
                shape: vec![2, 2],
                payload: f32_payload(&[0.0; 4]),
            },
            SynTensor {
                name: "layer.bias",
                dtype: "F32",
                shape: vec![2],
                payload: f32_payload(&[0.0; 2]),
            },
        ];
        let bytes = build_safetensors(&tensors, None);
        SafetensorsFile::from_bytes(&bytes).expect("load")
    }

    #[test]
    fn validate_happy_path() {
        let file = census_file();
        let manifest =
            WeightsManifest::new([("layer.weight", vec![2, 2]), ("layer.bias", vec![2])]);
        assert_eq!(manifest.len(), 2);
        assert!(!manifest.is_empty());
        validate(&file, &manifest).expect("census matches");
    }

    #[test]
    fn validate_reports_missing_and_mismatch_and_extra() {
        let file = census_file();
        // Declare a missing tensor and a wrong shape; do NOT declare
        // "layer.bias" so it shows up as an extra.
        let manifest = WeightsManifest::new([
            ("layer.weight", vec![4, 4]), // shape mismatch (file is [2,2])
            ("does.not.exist", vec![1]),  // missing
        ]);
        let err = validate(&file, &manifest).expect_err("should fail loud");
        let msg = err.to_string();
        assert!(msg.contains("MISSING"), "has missing section: {msg}");
        assert!(
            msg.contains("does.not.exist"),
            "names missing tensor: {msg}"
        );
        assert!(
            msg.contains("SHAPE MISMATCH"),
            "has mismatch section: {msg}"
        );
        assert!(
            msg.contains("layer.weight"),
            "names mismatched tensor: {msg}"
        );
        assert!(msg.contains("[4, 4]"), "shows expected shape: {msg}");
        assert!(msg.contains("[2, 2]"), "shows actual shape: {msg}");
        assert!(msg.contains("EXTRA"), "has extra section: {msg}");
        assert!(msg.contains("layer.bias"), "names extra tensor: {msg}");
    }

    #[test]
    fn validate_extras_only_still_fails() {
        let file = census_file();
        // Declare only one of the two file tensors at the right shape; the other
        // is an "extra" and must still trip the census check.
        let manifest = WeightsManifest::new([("layer.weight", vec![2, 2])]);
        let err = validate(&file, &manifest).expect_err("extras-only fails");
        let msg = err.to_string();
        assert!(msg.contains("EXTRA"), "msg: {msg}");
        assert!(msg.contains("layer.bias"), "msg: {msg}");
        assert!(!msg.contains("MISSING"), "no missing expected: {msg}");
    }

    // ─────────────────────────────────────────────────────────────────────
    // resolve_aux_model (no env mutation — explicit-dirs inner fn)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn resolve_aux_existing_path_canonicalized() {
        let dir = TempDir::new("auxpath");
        let path = write_file(dir.path(), "ecapa.safetensors", b"anything");
        let resolved = resolve_aux_model(path.to_str().expect("utf8")).expect("resolve");
        assert_eq!(resolved, path.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_aux_in_dirs_first_match_wins() {
        let high = TempDir::new("auxhi");
        let low = TempDir::new("auxlo");
        let hi = write_file(high.path(), "ecapa.safetensors", b"hi");
        let _lo = write_file(low.path(), "ecapa.safetensors", b"lo");
        let dirs = vec![high.path().to_path_buf(), low.path().to_path_buf()];
        let resolved = resolve_aux_model_in_dirs("ecapa.safetensors", &dirs).expect("resolve");
        assert_eq!(resolved, hi.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_aux_falls_through_to_later_dir() {
        let empty = TempDir::new("auxempty");
        let real = TempDir::new("auxreal");
        let path = write_file(real.path(), "sep.safetensors", b"x");
        let dirs = vec![empty.path().to_path_buf(), real.path().to_path_buf()];
        let resolved = resolve_aux_model_in_dirs("sep.safetensors", &dirs).expect("resolve");
        assert_eq!(resolved, path.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_aux_miss_error_lists_dirs() {
        let a = TempDir::new("auxa");
        let b = TempDir::new("auxb");
        let dirs = vec![a.path().to_path_buf(), b.path().to_path_buf()];
        let err = resolve_aux_model_in_dirs("nope.safetensors", &dirs).expect_err("should miss");
        let msg = err.to_string();
        assert!(msg.contains("nope.safetensors"), "names file: {msg}");
        assert!(
            msg.contains(&a.path().display().to_string()),
            "lists first dir: {msg}"
        );
        assert!(
            msg.contains(&b.path().display().to_string()),
            "lists second dir: {msg}"
        );
        assert!(msg.contains("fetch_aux_models.sh"), "actionable fix: {msg}");
        assert!(matches!(err, FwError::InvalidRequest(_)));
    }

    #[test]
    fn aux_search_dirs_include_aux_subdir_first() {
        // The expansion must place each base dir's `aux/` subdir immediately
        // before the base dir itself. We can't assert absolute paths (env
        // dependent) but we can assert the structural invariant on whatever
        // bases the current env produces, if any.
        let dirs = aux_search_dirs();
        // Walk in pairs: every even index should be `<base>/aux` whose parent
        // equals the following odd-index entry.
        for pair in dirs.chunks(2) {
            if pair.len() == 2 {
                assert_eq!(pair[0].file_name().and_then(|s| s.to_str()), Some("aux"));
                assert_eq!(pair[0].parent(), Some(pair[1].as_path()));
            }
        }
    }
}
