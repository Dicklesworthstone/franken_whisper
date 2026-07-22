use std::collections::HashMap;

use crate::model::{
    AccelerationBackend, AccelerationReport, TranscriptionResult, TranscriptionSegment,
};

pub fn apply(result: &mut TranscriptionResult) -> AccelerationReport {
    apply_with_token(result, None)
}

#[allow(clippy::vec_init_then_push)]
pub(crate) fn apply_with_token(
    result: &mut TranscriptionResult,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> AccelerationReport {
    if let Some(tok) = token
        && tok.checkpoint().is_err()
    {
        let report = AccelerationReport {
            backend: AccelerationBackend::None,
            input_values: 0,
            normalized_confidences: false,
            pre_mass: None,
            post_mass: None,
            notes: vec!["acceleration cancelled by pipeline checkpoint".to_owned()],
        };
        result.acceleration = Some(report.clone());
        return report;
    }
    tracing::debug!(stage = "acceleration", "Entering accelerate::apply");
    if result.segments.is_empty() {
        let report = AccelerationReport {
            backend: AccelerationBackend::None,
            input_values: 0,
            normalized_confidences: false,
            pre_mass: None,
            post_mass: None,
            notes: vec!["no segments available for acceleration".to_owned()],
        };
        result.acceleration = Some(report.clone());
        return report;
    }

    let baseline = confidence_vector(&result.segments);
    let pre_mass = Some(baseline.iter().copied().sum::<f64>());
    let mut notes = Vec::new();

    #[cfg(feature = "gpu-frankentorch")]
    {
        match normalize_with_frankentorch(&baseline) {
            Ok(values) => {
                let report = build_report(
                    result,
                    AccelerationBackend::Frankentorch,
                    values,
                    pre_mass,
                    notes,
                );
                result.acceleration = Some(report.clone());
                return report;
            }
            Err(error) => {
                notes.push(format!("frankentorch unavailable at runtime: {error}"));
            }
        }
    }

    #[cfg(feature = "gpu-frankenjax")]
    {
        match normalize_with_frankenjax(&baseline) {
            Ok(values) => {
                let report = build_report(
                    result,
                    AccelerationBackend::Frankenjax,
                    values,
                    pre_mass,
                    notes,
                );
                result.acceleration = Some(report.clone());
                return report;
            }
            Err(error) => {
                notes.push(format!("frankenjax unavailable at runtime: {error}"));
            }
        }
    }

    notes.push("using deterministic CPU normalization fallback".to_owned());
    let values = normalize_cpu(&baseline);
    let report = build_report(result, AccelerationBackend::None, values, pre_mass, notes);
    result.acceleration = Some(report.clone());
    report
}

fn build_report(
    result: &mut TranscriptionResult,
    backend: AccelerationBackend,
    values: Vec<f64>,
    pre_mass: Option<f64>,
    notes: Vec<String>,
) -> AccelerationReport {
    apply_confidences(&mut result.segments, &values);

    let post_mass = Some(values.iter().copied().sum::<f64>());
    AccelerationReport {
        backend,
        input_values: values.len(),
        normalized_confidences: true,
        pre_mass,
        post_mass,
        notes,
    }
}

fn confidence_vector(segments: &[TranscriptionSegment]) -> Vec<f64> {
    segments
        .iter()
        .map(|segment| {
            if let Some(confidence) = segment.confidence
                && confidence.is_finite()
                && confidence > 0.0
            {
                return confidence;
            }

            // Deterministic baseline when upstream backend omitted confidence.
            let text_weight = segment.text.chars().count().max(1) as f64;
            text_weight.ln_1p() + 1.0
        })
        .collect()
}

fn apply_confidences(segments: &mut [TranscriptionSegment], values: &[f64]) {
    for (segment, value) in segments.iter_mut().zip(values.iter().copied()) {
        // Sanitize before stamping: a NaN/Inf value would flow into the
        // downstream confidence clamps (whisper_cpp_native/decode) and panic on
        // the current nightly. Finite values keep their [0, 1] semantics.
        let v = if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
        segment.confidence = Some(v);
    }
}

fn normalize_cpu(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let safe_sum: f64 = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum();

    if safe_sum <= f64::EPSILON {
        return vec![1.0 / values.len() as f64; values.len()];
    }

    values
        .iter()
        .map(|value| {
            let safe = if value.is_finite() && *value > 0.0 {
                *value
            } else {
                0.0
            };
            safe / safe_sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// bd-1r7.1: Attention layer types and acceleration
// ---------------------------------------------------------------------------

/// Which kind of attention operation to accelerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    SelfAttention,
    CrossAttention,
}

/// Result of an attention layer acceleration pass.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionResult {
    pub kind: AttentionKind,
    pub scores: Vec<f64>,
    pub gpu_accelerated: bool,
}

/// Compute scaled dot-product attention scores on the CPU.
///
/// Given `query` and `key` vectors of equal length, produces a vector of
/// attention weights by computing element-wise product, scaling by
/// `1/sqrt(dim)`, and applying softmax normalization.
pub(crate) fn attention_scores_cpu(
    query: &[f64],
    key: &[f64],
    kind: AttentionKind,
) -> AttentionResult {
    if query.is_empty() || key.is_empty() {
        return AttentionResult {
            kind,
            scores: vec![],
            gpu_accelerated: false,
        };
    }

    let dim = query.len().min(key.len());
    let scale = 1.0 / (dim as f64).sqrt();

    // Compute raw attention logits (element-wise query*key scaled).
    let raw: Vec<f64> = query
        .iter()
        .zip(key.iter())
        .map(|(q, k)| {
            let q_safe = if q.is_finite() { *q } else { 0.0 };
            let k_safe = if k.is_finite() { *k } else { 0.0 };
            q_safe * k_safe * scale
        })
        .collect();

    // Apply softmax to get attention weights.
    let scores = softmax_cpu(&raw);

    AttentionResult {
        kind,
        scores,
        gpu_accelerated: false,
    }
}

/// GPU-accelerated attention scores via frankentorch.
#[cfg(feature = "gpu-frankentorch")]
pub(crate) fn attention_scores_frankentorch(
    query: &[f64],
    key: &[f64],
    kind: AttentionKind,
) -> Result<AttentionResult, String> {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let dim = query.len().min(key.len());

    let q_tensor = session
        .tensor_variable(query[..dim].to_vec(), vec![dim], false)
        .map_err(|e| e.to_string())?;
    let k_tensor = session
        .tensor_variable(key[..dim].to_vec(), vec![dim], false)
        .map_err(|e| e.to_string())?;

    let product = session
        .tensor_mul(q_tensor, k_tensor)
        .map_err(|e| e.to_string())?;

    let scale = 1.0 / (dim as f64).sqrt();
    let scale_tensor = session
        .tensor_variable(vec![scale; dim], vec![dim], false)
        .map_err(|e| e.to_string())?;
    let scaled = session
        .tensor_mul(product, scale_tensor)
        .map_err(|e| e.to_string())?;

    let softmaxed = session
        .tensor_softmax(scaled, 0)
        .map_err(|e| e.to_string())?;

    let scores = session
        .tensor_values(softmaxed)
        .map_err(|e| e.to_string())?;

    Ok(AttentionResult {
        kind,
        scores,
        gpu_accelerated: true,
    })
}

/// Fallback stub when `gpu-frankentorch` is not enabled.
#[cfg(not(feature = "gpu-frankentorch"))]
pub(crate) fn attention_scores_frankentorch(
    _query: &[f64],
    _key: &[f64],
    _kind: AttentionKind,
) -> Result<AttentionResult, String> {
    Err("gpu-frankentorch feature not enabled".to_owned())
}

/// Dispatch attention scoring: GPU if available, CPU fallback otherwise.
pub fn compute_attention(query: &[f64], key: &[f64], kind: AttentionKind) -> AttentionResult {
    match attention_scores_frankentorch(query, key, kind) {
        Ok(result) => result,
        Err(_) => attention_scores_cpu(query, key, kind),
    }
}

// ---------------------------------------------------------------------------
// bd-1r7.1: Embedding layer acceleration
// ---------------------------------------------------------------------------

/// Which kind of embedding to accelerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingKind {
    Token,
    Positional,
}

/// Result of an embedding lookup operation.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResult {
    pub kind: EmbeddingKind,
    pub embeddings: Vec<Vec<f64>>,
    pub gpu_accelerated: bool,
}

/// CPU embedding lookup: for each token index, retrieve the corresponding row
/// from the embedding table. Out-of-range indices produce a zero vector.
pub(crate) fn embedding_lookup_cpu(
    indices: &[usize],
    table: &[Vec<f64>],
    kind: EmbeddingKind,
) -> EmbeddingResult {
    let embed_dim = table.first().map_or(0, Vec::len);

    let embeddings: Vec<Vec<f64>> = indices
        .iter()
        .map(|&idx| {
            if idx < table.len() {
                table[idx]
                    .iter()
                    .map(|v| if v.is_finite() { *v } else { 0.0 })
                    .collect()
            } else {
                vec![0.0; embed_dim]
            }
        })
        .collect();

    EmbeddingResult {
        kind,
        embeddings,
        gpu_accelerated: false,
    }
}

/// GPU-accelerated embedding lookup via frankentorch.
#[cfg(feature = "gpu-frankentorch")]
pub(crate) fn embedding_lookup_frankentorch(
    indices: &[usize],
    table: &[Vec<f64>],
    kind: EmbeddingKind,
) -> Result<EmbeddingResult, String> {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;

    let embed_dim = table.first().map_or(0, Vec::len);
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

    // Flatten the embedding table into a single tensor.
    let flat: Vec<f64> = table.iter().flat_map(|row| row.iter().copied()).collect();
    let table_tensor = session
        .tensor_variable(flat, vec![table.len(), embed_dim], false)
        .map_err(|e| e.to_string())?;

    let index_floats: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
    let idx_tensor = session
        .tensor_variable(index_floats, vec![indices.len()], false)
        .map_err(|e| e.to_string())?;

    let gathered = session
        .tensor_gather(table_tensor, 0, idx_tensor)
        .map_err(|e| e.to_string())?;

    let raw = session.tensor_values(gathered).map_err(|e| e.to_string())?;

    // Reshape flat output back into per-token embedding vectors.
    let embeddings: Vec<Vec<f64>> = raw.chunks(embed_dim).map(|c| c.to_vec()).collect();

    Ok(EmbeddingResult {
        kind,
        embeddings,
        gpu_accelerated: true,
    })
}

/// Fallback stub when `gpu-frankentorch` is not enabled.
#[cfg(not(feature = "gpu-frankentorch"))]
pub(crate) fn embedding_lookup_frankentorch(
    _indices: &[usize],
    _table: &[Vec<f64>],
    _kind: EmbeddingKind,
) -> Result<EmbeddingResult, String> {
    Err("gpu-frankentorch feature not enabled".to_owned())
}

/// Dispatch embedding lookup: GPU if available, CPU fallback otherwise.
pub fn compute_embedding(
    indices: &[usize],
    table: &[Vec<f64>],
    kind: EmbeddingKind,
) -> EmbeddingResult {
    match embedding_lookup_frankentorch(indices, table, kind) {
        Ok(result) => result,
        Err(_) => embedding_lookup_cpu(indices, table, kind),
    }
}

// ---------------------------------------------------------------------------
// bd-1r7.1: VAD (Voice Activity Detection) scoring acceleration
// ---------------------------------------------------------------------------

/// Result of a VAD scoring pass.
#[derive(Debug, Clone, PartialEq)]
pub struct VadResult {
    /// Per-frame probability of voice activity, each in [0.0, 1.0].
    pub frame_scores: Vec<f64>,
    /// Overall voice activity ratio across all frames.
    pub activity_ratio: f64,
    pub gpu_accelerated: bool,
}

/// CPU VAD scoring: applies sigmoid to raw energy values to produce
/// per-frame voice activity probabilities.
pub(crate) fn vad_scores_cpu(energy_values: &[f64]) -> VadResult {
    if energy_values.is_empty() {
        return VadResult {
            frame_scores: vec![],
            activity_ratio: 0.0,
            gpu_accelerated: false,
        };
    }

    let frame_scores: Vec<f64> = energy_values
        .iter()
        .map(|&e| {
            if !e.is_finite() {
                return 0.0;
            }
            sigmoid(e)
        })
        .collect();

    let activity_ratio = if frame_scores.is_empty() {
        0.0
    } else {
        frame_scores.iter().copied().sum::<f64>() / frame_scores.len() as f64
    };

    VadResult {
        frame_scores,
        activity_ratio,
        gpu_accelerated: false,
    }
}

/// GPU-accelerated VAD scoring via frankentorch.
#[cfg(feature = "gpu-frankentorch")]
pub(crate) fn vad_scores_frankentorch(energy_values: &[f64]) -> Result<VadResult, String> {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let tensor = session
        .tensor_variable(energy_values.to_vec(), vec![energy_values.len()], false)
        .map_err(|e| e.to_string())?;
    let activated = session.tensor_sigmoid(tensor).map_err(|e| e.to_string())?;

    let frame_scores = session
        .tensor_values(activated)
        .map_err(|e| e.to_string())?;

    let activity_ratio = if frame_scores.is_empty() {
        0.0
    } else {
        frame_scores.iter().copied().sum::<f64>() / frame_scores.len() as f64
    };

    Ok(VadResult {
        frame_scores,
        activity_ratio,
        gpu_accelerated: true,
    })
}

/// Fallback stub when `gpu-frankentorch` is not enabled.
#[cfg(not(feature = "gpu-frankentorch"))]
pub(crate) fn vad_scores_frankentorch(_energy_values: &[f64]) -> Result<VadResult, String> {
    Err("gpu-frankentorch feature not enabled".to_owned())
}

/// Dispatch VAD scoring: GPU if available, CPU fallback otherwise.
pub fn compute_vad_scores(energy_values: &[f64]) -> VadResult {
    match vad_scores_frankentorch(energy_values) {
        Ok(result) => result,
        Err(_) => vad_scores_cpu(energy_values),
    }
}

// ---------------------------------------------------------------------------
// bd-1r7.1: Layer normalization acceleration
// ---------------------------------------------------------------------------

/// Result of a layer normalization pass.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNormResult {
    /// The normalized output values.
    pub normalized: Vec<f64>,
    /// Whether GPU acceleration was used.
    pub gpu_accelerated: bool,
}

/// CPU layer normalization: subtracts mean and divides by standard deviation
/// (with a small epsilon for numerical stability), then applies scale (gamma)
/// and shift (beta) per element.
pub(crate) fn layer_norm_cpu(
    values: &[f64],
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
) -> LayerNormResult {
    if values.is_empty() {
        return LayerNormResult {
            normalized: vec![],
            gpu_accelerated: false,
        };
    }

    let n = values.len() as f64;

    // Compute mean over finite values, treating non-finite as 0.0.
    let sanitized: Vec<f64> = values
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();
    let mean = sanitized.iter().sum::<f64>() / n;

    // Compute variance.
    let variance = sanitized.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

    let epsilon = if epsilon.is_finite() {
        epsilon.max(f64::EPSILON)
    } else {
        f64::EPSILON
    };
    let std_dev = (variance + epsilon).sqrt();

    let normalized: Vec<f64> = sanitized
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let normed = (v - mean) / std_dev;
            let g = gamma.get(i).copied().unwrap_or(1.0);
            let b = beta.get(i).copied().unwrap_or(0.0);
            let g_safe = if g.is_finite() { g } else { 1.0 };
            let b_safe = if b.is_finite() { b } else { 0.0 };
            normed * g_safe + b_safe
        })
        .collect();

    LayerNormResult {
        normalized,
        gpu_accelerated: false,
    }
}

/// GPU-accelerated layer normalization via frankentorch.
#[cfg(feature = "gpu-frankentorch")]
pub(crate) fn layer_norm_frankentorch(
    _values: &[f64],
    _gamma: &[f64],
    _beta: &[f64],
    _epsilon: f64,
) -> Result<LayerNormResult, String> {
    Err("frankentorch layer-norm op unavailable in current ft-api; using CPU fallback".to_owned())
}

/// Fallback stub when `gpu-frankentorch` is not enabled.
#[cfg(not(feature = "gpu-frankentorch"))]
pub(crate) fn layer_norm_frankentorch(
    _values: &[f64],
    _gamma: &[f64],
    _beta: &[f64],
    _epsilon: f64,
) -> Result<LayerNormResult, String> {
    Err("gpu-frankentorch feature not enabled".to_owned())
}

/// Dispatch layer normalization: GPU if available, CPU fallback otherwise.
pub fn compute_layer_norm(
    values: &[f64],
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
) -> LayerNormResult {
    match layer_norm_frankentorch(values, gamma, beta, epsilon) {
        Ok(result) => result,
        Err(_) => layer_norm_cpu(values, gamma, beta, epsilon),
    }
}

// ---------------------------------------------------------------------------
// bd-1r7.2: JIT-compiled inference paths (frankenjax)
// ---------------------------------------------------------------------------

/// Describes a graph pattern that can be JIT-compiled for inference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferenceGraphPattern {
    /// Normalize-then-softmax pipeline for confidence scoring.
    NormalizeSoftmax,
    /// Linear projection followed by activation (e.g. for feed-forward layers).
    LinearActivation,
    /// Full attention pattern: Q*K^T scaling + softmax + V projection.
    AttentionBlock,
}

/// A cached compiled kernel for a specific graph pattern.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub pattern: InferenceGraphPattern,
    pub kernel_id: String,
}

/// Cache for JIT-compiled kernels. Maps graph patterns to compiled
/// kernel descriptors so repeated inference calls avoid recompilation.
#[derive(Debug, Clone)]
pub struct JitKernelCache {
    entries: HashMap<InferenceGraphPattern, CompiledKernel>,
}

impl JitKernelCache {
    /// Create a new empty kernel cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Return the number of cached kernels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a compiled kernel for the given pattern.
    #[must_use]
    pub fn get(&self, pattern: &InferenceGraphPattern) -> Option<&CompiledKernel> {
        self.entries.get(pattern)
    }

    /// Insert a compiled kernel into the cache.
    pub fn insert(&mut self, kernel: CompiledKernel) {
        self.entries.insert(kernel.pattern.clone(), kernel);
    }

    /// Clear all cached kernels.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for JitKernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a JIT-compiled inference pass.
#[derive(Debug, Clone, PartialEq)]
pub struct JitInferenceResult {
    pub values: Vec<f64>,
    pub pattern: InferenceGraphPattern,
    pub cache_hit: bool,
    pub gpu_accelerated: bool,
}

/// CPU fallback for normalize-softmax inference.
pub(crate) fn jit_normalize_softmax_cpu(values: &[f64]) -> Vec<f64> {
    let normalized = normalize_cpu(values);
    softmax_cpu(&normalized)
}

/// CPU fallback for linear-activation inference.
/// Applies a simple linear transform (weights * inputs + bias) followed by ReLU.
pub(crate) fn jit_linear_activation_cpu(inputs: &[f64], weights: &[f64], bias: f64) -> Vec<f64> {
    inputs
        .iter()
        .zip(weights.iter())
        .map(|(x, w)| {
            let x_safe = if x.is_finite() { *x } else { 0.0 };
            let w_safe = if w.is_finite() { *w } else { 0.0 };
            let linear = x_safe * w_safe + bias;
            // ReLU activation
            linear.max(0.0)
        })
        .collect()
}

/// CPU fallback for attention-block inference.
/// Computes Q*K scaling, softmax, then multiplies by V.
pub(crate) fn jit_attention_block_cpu(query: &[f64], key: &[f64], value: &[f64]) -> Vec<f64> {
    if query.is_empty() || key.is_empty() || value.is_empty() {
        return vec![];
    }

    let dim = query.len().min(key.len()).min(value.len());
    let scale = 1.0 / (dim as f64).sqrt();

    let logits: Vec<f64> = query[..dim]
        .iter()
        .zip(key[..dim].iter())
        .map(|(q, k)| {
            let q_safe = if q.is_finite() { *q } else { 0.0 };
            let k_safe = if k.is_finite() { *k } else { 0.0 };
            q_safe * k_safe * scale
        })
        .collect();

    let weights = softmax_cpu(&logits);

    // Multiply attention weights by value vector element-wise.
    weights
        .iter()
        .zip(value[..dim].iter())
        .map(|(w, v)| {
            let v_safe = if v.is_finite() { *v } else { 0.0 };
            w * v_safe
        })
        .collect()
}

/// GPU-accelerated JIT inference via frankenjax.
#[cfg(feature = "gpu-frankenjax")]
pub(crate) fn jit_inference_frankenjax(
    pattern: &InferenceGraphPattern,
    inputs: &[&[f64]],
    cache: &mut JitKernelCache,
) -> Result<JitInferenceResult, String> {
    let cache_hit = cache.get(pattern).is_some();

    if !cache_hit {
        // Compile and cache the kernel for this pattern.
        let kernel_id = match pattern {
            InferenceGraphPattern::NormalizeSoftmax => "jax_norm_softmax_v1",
            InferenceGraphPattern::LinearActivation => "jax_linear_act_v1",
            InferenceGraphPattern::AttentionBlock => "jax_attn_block_v1",
        };
        cache.insert(CompiledKernel {
            pattern: pattern.clone(),
            kernel_id: kernel_id.to_owned(),
        });
    }

    match pattern {
        InferenceGraphPattern::NormalizeSoftmax => {
            let data = inputs.first().ok_or("missing input for NormalizeSoftmax")?;
            let normalized = normalize_with_frankenjax(data)?;
            let values = softmax_cpu(&normalized);

            Ok(JitInferenceResult {
                values,
                pattern: pattern.clone(),
                cache_hit,
                gpu_accelerated: true,
            })
        }
        InferenceGraphPattern::LinearActivation => Err(
            "frankenjax ProgramSpec no longer exposes LinearActivation kernel; CPU fallback is used"
                .to_owned(),
        ),
        InferenceGraphPattern::AttentionBlock => Err(
            "frankenjax ProgramSpec no longer exposes AttentionBlock kernel; CPU fallback is used"
                .to_owned(),
        ),
    }
}

/// Run JIT-compiled inference: GPU (frankenjax) if available, CPU fallback otherwise.
///
/// For `NormalizeSoftmax`: expects `inputs[0]` = data vector.
/// For `LinearActivation`: expects `inputs[0]` = data, `inputs[1]` = weights.
///   The bias is assumed to be 0.0 in the dispatch (caller can offset externally).
/// For `AttentionBlock`: expects `inputs[0]` = query, `inputs[1]` = key, `inputs[2]` = value.
pub fn jit_inference(
    pattern: &InferenceGraphPattern,
    inputs: &[&[f64]],
    cache: &mut JitKernelCache,
) -> JitInferenceResult {
    #[cfg(feature = "gpu-frankenjax")]
    {
        match jit_inference_frankenjax(pattern, inputs, cache) {
            Ok(result) => return result,
            Err(_) => {}
        }
    }

    let cache_hit = cache.get(pattern).is_some();

    // Ensure the pattern is "compiled" in the cache even for CPU fallback.
    if !cache_hit {
        let kernel_id = match pattern {
            InferenceGraphPattern::NormalizeSoftmax => "cpu_norm_softmax_v1",
            InferenceGraphPattern::LinearActivation => "cpu_linear_act_v1",
            InferenceGraphPattern::AttentionBlock => "cpu_attn_block_v1",
        };
        cache.insert(CompiledKernel {
            pattern: pattern.clone(),
            kernel_id: kernel_id.to_owned(),
        });
    }

    let values = match pattern {
        InferenceGraphPattern::NormalizeSoftmax => {
            let data = inputs.first().copied().unwrap_or(&[]);
            jit_normalize_softmax_cpu(data)
        }
        InferenceGraphPattern::LinearActivation => {
            let data = inputs.first().copied().unwrap_or(&[]);
            let weights = inputs.get(1).copied().unwrap_or(&[]);
            jit_linear_activation_cpu(data, weights, 0.0)
        }
        InferenceGraphPattern::AttentionBlock => {
            let q = inputs.first().copied().unwrap_or(&[]);
            let k = inputs.get(1).copied().unwrap_or(&[]);
            let v = inputs.get(2).copied().unwrap_or(&[]);
            jit_attention_block_cpu(q, k, v)
        }
    };

    // Use cache_hit from *before* we inserted above for the first call.
    let was_cached = cache.get(pattern).is_some() && cache_hit;

    JitInferenceResult {
        values,
        pattern: pattern.clone(),
        cache_hit: was_cached,
        gpu_accelerated: false,
    }
}

// ---------------------------------------------------------------------------
// bd-1r7.2: InferenceGraphSpec and JIT compilation/execution helpers
// ---------------------------------------------------------------------------

/// Describes a computation graph that can be JIT-compiled for inference.
///
/// This captures the shape and pattern of the computation so that the JIT
/// compiler can produce an optimized kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceGraphSpec {
    /// The computation pattern to compile.
    pub pattern: InferenceGraphPattern,
    /// Number of input tensors expected by this graph.
    pub input_count: usize,
    /// Dimension of each input tensor (used for shape validation).
    pub input_dim: usize,
    /// Optional human-readable label for debugging/logging.
    pub label: Option<String>,
}

impl InferenceGraphSpec {
    /// Create a new graph spec with the given pattern, input count, and dimension.
    #[must_use]
    pub fn new(pattern: InferenceGraphPattern, input_count: usize, input_dim: usize) -> Self {
        Self {
            pattern,
            input_count,
            input_dim,
            label: None,
        }
    }

    /// Attach a human-readable label for debugging.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Compute a deterministic cache key for a compiled graph based on its spec.
///
/// The key is a string that uniquely identifies the spec's pattern, input
/// count, and input dimension so that identical graph shapes share the same
/// compiled kernel.
#[must_use]
pub fn jit_cache_key(spec: &InferenceGraphSpec) -> String {
    format!(
        "{:?}:inputs={}:dim={}",
        spec.pattern, spec.input_count, spec.input_dim
    )
}

/// Compile an inference graph from a spec, returning a `CompiledKernel`.
///
/// When the `gpu-frankenjax` feature is enabled this delegates to the JAX
/// JIT compiler; otherwise a deterministic CPU kernel descriptor is produced.
#[cfg(feature = "gpu-frankenjax")]
pub fn jit_compile_inference_graph(
    spec: &InferenceGraphSpec,
    cache: &mut JitKernelCache,
) -> Result<CompiledKernel, String> {
    let key = jit_cache_key(spec);

    // Return cached kernel if available.
    if let Some(existing) = cache.get(&spec.pattern) {
        return Ok(existing.clone());
    }

    // Compile via frankenjax (simplified: we produce a kernel descriptor).
    let kernel_id = format!("jax_jit_{key}");
    let kernel = CompiledKernel {
        pattern: spec.pattern.clone(),
        kernel_id,
    };
    cache.insert(kernel.clone());
    Ok(kernel)
}

/// Fallback stub when `gpu-frankenjax` is not enabled.
#[cfg(not(feature = "gpu-frankenjax"))]
pub fn jit_compile_inference_graph(
    spec: &InferenceGraphSpec,
    cache: &mut JitKernelCache,
) -> Result<CompiledKernel, String> {
    let key = jit_cache_key(spec);

    if let Some(existing) = cache.get(&spec.pattern) {
        return Ok(existing.clone());
    }

    let kernel_id = format!("cpu_jit_{key}");
    let kernel = CompiledKernel {
        pattern: spec.pattern.clone(),
        kernel_id,
    };
    cache.insert(kernel.clone());
    Ok(kernel)
}

/// Execute a batch of input vectors through a JIT-compiled inference graph.
///
/// Each element of `batch` is a set of input slices matching the graph spec.
/// Returns one `JitInferenceResult` per batch element.
///
/// When the `gpu-frankenjax` feature is enabled, this attempts GPU execution
/// for each batch element before falling back to CPU.
#[cfg(feature = "gpu-frankenjax")]
pub fn jit_execute_batch(
    spec: &InferenceGraphSpec,
    batch: &[Vec<Vec<f64>>],
    cache: &mut JitKernelCache,
) -> Vec<JitInferenceResult> {
    // Ensure the graph is compiled.
    let _ = jit_compile_inference_graph(spec, cache);

    batch
        .iter()
        .map(|inputs| {
            let slices: Vec<&[f64]> = inputs.iter().map(Vec::as_slice).collect();
            jit_inference(&spec.pattern, &slices, cache)
        })
        .collect()
}

/// Fallback stub when `gpu-frankenjax` is not enabled.
#[cfg(not(feature = "gpu-frankenjax"))]
pub fn jit_execute_batch(
    spec: &InferenceGraphSpec,
    batch: &[Vec<Vec<f64>>],
    cache: &mut JitKernelCache,
) -> Vec<JitInferenceResult> {
    // Ensure the graph is compiled (CPU path).
    let _ = jit_compile_inference_graph(spec, cache);

    batch
        .iter()
        .map(|inputs| {
            let slices: Vec<&[f64]> = inputs.iter().map(Vec::as_slice).collect();
            jit_inference(&spec.pattern, &slices, cache)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shared math utilities
// ---------------------------------------------------------------------------

/// Numerically stable softmax over a slice of f64 values.
pub(crate) fn softmax_cpu(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let max_val = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let max_val = if max_val.is_finite() { max_val } else { 0.0 };

    let exps: Vec<f64> = values
        .iter()
        .map(|v| {
            if v.is_finite() {
                (v - max_val).exp()
            } else {
                0.0
            }
        })
        .collect();

    let sum: f64 = exps.iter().sum();

    if sum <= f64::EPSILON {
        return vec![1.0 / values.len() as f64; values.len()];
    }

    exps.iter().map(|e| e / sum).collect()
}

/// Standard sigmoid function, clamped for numerical safety.
pub(crate) fn sigmoid(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    // Clamp to avoid overflow in exp.
    let clamped = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-clamped).exp())
}

#[cfg(feature = "gpu-frankentorch")]
fn normalize_with_frankentorch(values: &[f64]) -> Result<Vec<f64>, String> {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let tensor = session
        .tensor_variable(values.to_vec(), vec![values.len()], false)
        .map_err(|error| error.to_string())?;
    let normalized = session
        .tensor_softmax(tensor, 0)
        .map_err(|error| error.to_string())?;

    let out = session
        .tensor_values(normalized)
        .map_err(|error| error.to_string())?;

    // GPU softmax can emit NaN/Inf (all-(-inf) inputs, overflow, or a NaN in
    // values). A non-finite/degenerate result would be stamped onto segment
    // confidences and panic the downstream clamps, so fall back to the
    // defensive CPU path when the tensor output is not a usable distribution.
    if out.iter().all(|v| v.is_finite()) && out.iter().sum::<f64>() > f64::EPSILON {
        Ok(out)
    } else {
        Ok(normalize_cpu(values))
    }
}

#[cfg(feature = "gpu-frankenjax")]
fn normalize_with_frankenjax(values: &[f64]) -> Result<Vec<f64>, String> {
    use fj_api::jit;
    use fj_core::{ProgramSpec, Value, build_program};

    let vector = Value::vector_f64(values).map_err(|error| error.to_string())?;
    let result = jit(build_program(ProgramSpec::ReduceSumVec))
        .call(vec![vector])
        .map_err(|error| format!("jit reduce failed: {error}"))?;

    let total = result
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or_else(|| "reduce output did not contain scalar".to_owned())?;

    // Mirror normalize_cpu's defenses: the raw ReduceSumVec total does no finite
    // filtering, so a NaN/Inf input (or the denormal 2.225e-308) yields a
    // non-finite or vanishing total. `total <= f64::EPSILON` is false for NaN,
    // so guard finiteness explicitly and fall back to a uniform distribution.
    if !total.is_finite() || total <= f64::EPSILON {
        return Ok(vec![1.0 / values.len() as f64; values.len()]);
    }

    Ok(values
        .iter()
        .map(|value| {
            let safe = if value.is_finite() && *value > 0.0 {
                *value
            } else {
                0.0
            };
            safe / total
        })
        .collect())
}

// ---------------------------------------------------------------------------
// bd-1r7.3: CPU/GPU benchmark harness for acceleration decisions
// ---------------------------------------------------------------------------

/// Recommendation for which compute backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AccelRecommendation {
    /// CPU is the clear winner (or GPU is unavailable).
    UseCpu,
    /// GPU is the clear winner.
    UseGpu,
    /// Difference is negligible (<10% speedup either way).
    EitherFine,
}

impl std::fmt::Display for AccelRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UseCpu => write!(f, "use_cpu"),
            Self::UseGpu => write!(f, "use_gpu"),
            Self::EitherFine => write!(f, "either_fine"),
        }
    }
}

/// Result of a single benchmark comparison between CPU and GPU paths.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkResult {
    /// Name of the operation benchmarked (e.g. "softmax", "layer_norm").
    pub operation: String,
    /// CPU execution time in microseconds.
    pub cpu_time_us: u64,
    /// GPU execution time in microseconds, or `None` if GPU is unavailable.
    pub gpu_time_us: Option<u64>,
    /// Speedup ratio (cpu_time / gpu_time). Set to 0.0 when GPU is unavailable.
    pub speedup_ratio: f64,
    /// Which backend to use based on this benchmark.
    pub recommendation: AccelRecommendation,
}

/// Serializable report aggregating all benchmark results (robot mode output).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkReport {
    /// Individual benchmark results.
    pub results: Vec<BenchmarkResult>,
    /// Overall recommendation aggregated from all benchmarks.
    pub overall: AccelRecommendation,
    /// Human-readable notes about the benchmark run.
    pub notes: Vec<String>,
}

/// Harness that measures execution time of compute operations on CPU vs GPU
/// to inform acceleration decisions.
pub struct BenchmarkHarness {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkHarness {
    /// Create a new, empty benchmark harness.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Benchmark softmax computation at the given size.
    ///
    /// Times the CPU path via [`softmax_cpu`]. The GPU path returns `None`
    /// since GPU is not actually available in the current build.
    pub fn benchmark_softmax(&mut self, size: usize) -> BenchmarkResult {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();

        let start = std::time::Instant::now();
        let _cpu_result = softmax_cpu(&values);
        let cpu_elapsed = start.elapsed();
        let cpu_time_us = cpu_elapsed.as_micros() as u64;

        // GPU path: not available, so gpu_time_us is None.
        let gpu_time_us: Option<u64> = None;
        let speedup_ratio = 0.0;
        let recommendation = AccelRecommendation::UseCpu;

        let result = BenchmarkResult {
            operation: format!("softmax(size={})", size),
            cpu_time_us,
            gpu_time_us,
            speedup_ratio,
            recommendation,
        };
        self.results.push(result.clone());
        result
    }

    /// Benchmark layer normalization at the given size.
    ///
    /// Times the CPU path via [`layer_norm_cpu`]. The GPU path returns `None`
    /// since GPU is not actually available in the current build.
    pub fn benchmark_layer_norm(&mut self, size: usize) -> BenchmarkResult {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let gamma: Vec<f64> = vec![1.0; size];
        let beta: Vec<f64> = vec![0.0; size];
        let epsilon = 1e-5;

        let start = std::time::Instant::now();
        let _cpu_result = layer_norm_cpu(&values, &gamma, &beta, epsilon);
        let cpu_elapsed = start.elapsed();
        let cpu_time_us = cpu_elapsed.as_micros() as u64;

        // GPU path: not available.
        let gpu_time_us: Option<u64> = None;
        let speedup_ratio = 0.0;
        let recommendation = AccelRecommendation::UseCpu;

        let result = BenchmarkResult {
            operation: format!("layer_norm(size={})", size),
            cpu_time_us,
            gpu_time_us,
            speedup_ratio,
            recommendation,
        };
        self.results.push(result.clone());
        result
    }

    /// Benchmark attention computation for a given sequence length and model
    /// dimension.
    ///
    /// Times the CPU path via [`attention_scores_cpu`]. The GPU path returns
    /// `None` since GPU is not actually available in the current build.
    pub fn benchmark_attention(&mut self, seq_len: usize, d_model: usize) -> BenchmarkResult {
        let dim = seq_len.max(d_model);
        let query: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.02).collect();
        let key: Vec<f64> = (0..dim).map(|i| ((i + 1) as f64) * 0.03).collect();

        let start = std::time::Instant::now();
        let _cpu_result = attention_scores_cpu(&query, &key, AttentionKind::SelfAttention);
        let cpu_elapsed = start.elapsed();
        let cpu_time_us = cpu_elapsed.as_micros() as u64;

        // GPU path: not available.
        let gpu_time_us: Option<u64> = None;
        let speedup_ratio = 0.0;
        let recommendation = AccelRecommendation::UseCpu;

        let result = BenchmarkResult {
            operation: format!("attention(seq_len={}, d_model={})", seq_len, d_model),
            cpu_time_us,
            gpu_time_us,
            speedup_ratio,
            recommendation,
        };
        self.results.push(result.clone());
        result
    }

    /// Run all benchmarks with representative sizes and return the full set
    /// of results.
    #[must_use]
    pub fn run_all(&mut self) -> Vec<BenchmarkResult> {
        self.results.clear();

        // Softmax at several sizes.
        self.benchmark_softmax(64);
        self.benchmark_softmax(512);
        self.benchmark_softmax(4096);

        // Layer norm at several sizes.
        self.benchmark_layer_norm(64);
        self.benchmark_layer_norm(512);
        self.benchmark_layer_norm(4096);

        // Attention at representative configs.
        self.benchmark_attention(128, 64);
        self.benchmark_attention(512, 256);

        self.results.clone()
    }

    /// Aggregate all benchmark results into a single overall recommendation.
    ///
    /// Since GPU is currently unavailable, this always returns `UseCpu`.
    /// When GPU becomes available, this will count votes across benchmarks
    /// and return the majority recommendation, with `EitherFine` used when
    /// the spread is within 10%.
    #[must_use]
    pub fn recommend_backend(&self) -> AccelRecommendation {
        if self.results.is_empty() {
            return AccelRecommendation::UseCpu;
        }

        let mut cpu_votes: u32 = 0;
        let mut gpu_votes: u32 = 0;
        let mut either_votes: u32 = 0;

        for result in &self.results {
            match result.recommendation {
                AccelRecommendation::UseCpu => cpu_votes += 1,
                AccelRecommendation::UseGpu => gpu_votes += 1,
                AccelRecommendation::EitherFine => either_votes += 1,
            }
        }

        if gpu_votes > cpu_votes + either_votes {
            AccelRecommendation::UseGpu
        } else if cpu_votes > gpu_votes + either_votes {
            AccelRecommendation::UseCpu
        } else {
            // When evenly split or mostly "either fine", call it fine.
            AccelRecommendation::EitherFine
        }
    }

    /// Generate a serializable benchmark report for robot mode output.
    #[must_use]
    pub fn report(&self) -> BenchmarkReport {
        let overall = self.recommend_backend();
        let mut notes = Vec::new();

        let gpu_available = self.results.iter().any(|r| r.gpu_time_us.is_some());
        if !gpu_available {
            notes.push(
                "GPU backend not available; all benchmarks ran CPU-only. \
                 GPU times are reported as null."
                    .to_owned(),
            );
        }

        BenchmarkReport {
            results: self.results.clone(),
            overall,
            notes,
        }
    }
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: compute a [`BenchmarkResult`] from CPU and optional GPU timings.
///
/// When both timings are available, computes the speedup ratio and picks the
/// appropriate recommendation using the 10% threshold for [`AccelRecommendation::EitherFine`].
#[must_use]
pub fn benchmark_result_from_timings(
    operation: String,
    cpu_time_us: u64,
    gpu_time_us: Option<u64>,
) -> BenchmarkResult {
    match gpu_time_us {
        Some(gpu_us) if gpu_us > 0 => {
            let ratio = cpu_time_us as f64 / gpu_us as f64;
            let recommendation = if ratio > 1.1 {
                // GPU is >10% faster than CPU.
                AccelRecommendation::UseGpu
            } else if ratio < 0.9 {
                // CPU is >10% faster than GPU.
                AccelRecommendation::UseCpu
            } else {
                AccelRecommendation::EitherFine
            };
            BenchmarkResult {
                operation,
                cpu_time_us,
                gpu_time_us: Some(gpu_us),
                speedup_ratio: ratio,
                recommendation,
            }
        }
        _ => {
            // GPU not available or zero timing — recommend CPU.
            BenchmarkResult {
                operation,
                cpu_time_us,
                gpu_time_us,
                speedup_ratio: 0.0,
                recommendation: AccelRecommendation::UseCpu,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::model::{
        AccelerationBackend, BackendKind, TranscriptionResult, TranscriptionSegment,
    };

    use super::apply;

    #[test]
    fn cpu_normalization_populates_confidences() {
        let mut result = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "a b c".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.0),
                    text: "alpha".to_owned(),
                    speaker: None,
                    confidence: None,
                },
                TranscriptionSegment {
                    start_sec: Some(1.0),
                    end_sec: Some(2.0),
                    text: "beta".to_owned(),
                    speaker: None,
                    confidence: Some(2.0),
                },
            ],
            acceleration: None,
            raw_output: json!({}),
            artifact_paths: vec![],
        };

        let report = apply(&mut result);
        let sum = result
            .segments
            .iter()
            .map(|segment| segment.confidence.unwrap_or_default())
            .sum::<f64>();

        assert!(report.normalized_confidences);
        assert_eq!(report.input_values, 2);
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(
            result.acceleration.as_ref().map(|meta| meta.backend),
            Some(report.backend)
        );
    }

    #[test]
    fn no_segments_short_circuit() {
        let mut result = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: String::new(),
            language: None,
            segments: vec![],
            acceleration: None,
            raw_output: json!({}),
            artifact_paths: vec![],
        };

        let report = apply(&mut result);
        assert_eq!(report.input_values, 0);
        assert!(!report.normalized_confidences);
        assert_eq!(report.backend, crate::model::AccelerationBackend::None);
    }

    #[test]
    fn backend_priority_prefers_frankentorch_then_frankenjax() {
        let priority: Vec<AccelerationBackend> = vec![
            #[cfg(feature = "gpu-frankentorch")]
            AccelerationBackend::Frankentorch,
            #[cfg(feature = "gpu-frankenjax")]
            AccelerationBackend::Frankenjax,
        ];

        if priority.len() == 2 {
            assert_eq!(
                priority,
                vec![
                    AccelerationBackend::Frankentorch,
                    AccelerationBackend::Frankenjax
                ]
            );
        }
    }

    // --- Edge-case tests for acceleration hardening ---

    fn make_result(segments: Vec<TranscriptionSegment>) -> TranscriptionResult {
        TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "test".to_owned(),
            language: None,
            segments,
            acceleration: None,
            raw_output: json!({}),
            artifact_paths: vec![],
        }
    }

    fn seg(text: &str, confidence: Option<f64>) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: text.to_owned(),
            speaker: None,
            confidence,
        }
    }

    #[test]
    fn all_zero_confidences_produce_uniform_normalization() {
        let mut result = make_result(vec![
            seg("a", Some(0.0)),
            seg("b", Some(0.0)),
            seg("c", Some(0.0)),
        ]);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);
        assert_eq!(report.input_values, 3);

        // Zero confidences → all use text-weight baseline → normalized
        let sum: f64 = result.segments.iter().filter_map(|s| s.confidence).sum();
        assert!((sum - 1.0).abs() < 1e-6, "should sum to 1.0 but got {sum}");
    }

    #[test]
    fn negative_confidences_treated_as_baseline() {
        let mut result = make_result(vec![seg("hello", Some(-0.5)), seg("world", Some(-1.0))]);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);

        // Negative confidences should be replaced by text-weight baseline
        for seg in &result.segments {
            assert!(
                seg.confidence.unwrap_or(0.0) >= 0.0,
                "normalized confidence should be non-negative"
            );
        }
        let sum: f64 = result.segments.iter().filter_map(|s| s.confidence).sum();
        assert!((sum - 1.0).abs() < 1e-6, "should sum to 1.0 but got {sum}");
    }

    #[test]
    fn nan_confidences_treated_as_baseline() {
        let mut result = make_result(vec![
            seg("nan test", Some(f64::NAN)),
            seg("valid", Some(0.8)),
        ]);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);

        for seg in &result.segments {
            let c = seg.confidence.unwrap_or(0.0);
            assert!(c.is_finite(), "confidence should be finite, got {c}");
            assert!(c >= 0.0, "confidence should be non-negative, got {c}");
        }
    }

    #[test]
    fn infinity_confidences_treated_as_baseline() {
        let mut result = make_result(vec![
            seg("inf test", Some(f64::INFINITY)),
            seg("valid", Some(0.5)),
        ]);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);

        for seg in &result.segments {
            let c = seg.confidence.unwrap_or(0.0);
            assert!(c.is_finite(), "confidence should be finite, got {c}");
        }
    }

    #[test]
    fn single_segment_normalizes_to_one() {
        let mut result = make_result(vec![seg("solo", Some(0.7))]);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);
        assert_eq!(report.input_values, 1);

        let c = result.segments[0].confidence.unwrap();
        assert!(
            (c - 1.0).abs() < 1e-6,
            "single segment should normalize to 1.0, got {c}"
        );
    }

    #[test]
    fn many_segments_normalize_correctly() {
        let segments: Vec<TranscriptionSegment> = (0..500)
            .map(|i| seg(&format!("seg-{i}"), Some((i as f64 + 1.0) / 500.0)))
            .collect();
        let mut result = make_result(segments);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);
        assert_eq!(report.input_values, 500);

        let sum: f64 = result.segments.iter().filter_map(|s| s.confidence).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "500 segments should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn mixed_none_and_valid_confidences() {
        let mut result = make_result(vec![
            seg("no conf", None),
            seg("has conf", Some(0.9)),
            seg("also none", None),
        ]);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);
        assert_eq!(report.input_values, 3);

        // All segments should now have Some confidence
        for seg in &result.segments {
            assert!(seg.confidence.is_some());
        }

        let sum: f64 = result.segments.iter().filter_map(|s| s.confidence).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "mixed confidences should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn acceleration_report_has_pre_and_post_mass() {
        let mut result = make_result(vec![seg("a", Some(0.5)), seg("b", Some(0.3))]);

        let report = apply(&mut result);
        assert!(report.pre_mass.is_some());
        assert!(report.post_mass.is_some());

        let pre = report.pre_mass.unwrap();
        assert!((pre - 0.8).abs() < 1e-6, "pre_mass should be 0.8");

        let post = report.post_mass.unwrap();
        assert!(
            (post - 1.0).abs() < 1e-6,
            "post_mass should be 1.0 (normalized)"
        );
    }

    #[test]
    fn cpu_normalize_all_non_finite_gives_uniform() {
        use super::normalize_cpu;
        let values = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let result = normalize_cpu(&values);
        assert_eq!(result.len(), 3);
        for v in &result {
            assert!(
                (*v - 1.0 / 3.0).abs() < 1e-6,
                "should be uniform 1/3, got {v}"
            );
        }
    }

    // --- Direct confidence_vector tests ---

    #[test]
    fn confidence_vector_uses_existing_positive_finite_values() {
        use super::confidence_vector;
        let segments = vec![seg("hello", Some(0.9)), seg("world", Some(0.7))];
        let vec = confidence_vector(&segments);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 0.9);
        assert_eq!(vec[1], 0.7);
    }

    #[test]
    fn confidence_vector_replaces_none_with_text_baseline() {
        use super::confidence_vector;
        let segments = vec![seg("hello", None)];
        let vec = confidence_vector(&segments);
        assert_eq!(vec.len(), 1);
        // text_weight = "hello".chars().count().max(1) = 5
        // baseline = 5.0_f64.ln_1p() + 1.0 ≈ 2.7918
        let expected = 5.0_f64.ln_1p() + 1.0;
        assert!(
            (vec[0] - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            vec[0]
        );
    }

    #[test]
    fn confidence_vector_replaces_zero_with_text_baseline() {
        use super::confidence_vector;
        let segments = vec![seg("ab", Some(0.0))];
        let vec = confidence_vector(&segments);
        // 0.0 is not > 0.0, so uses text baseline
        let expected = 2.0_f64.ln_1p() + 1.0;
        assert!(
            (vec[0] - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            vec[0]
        );
    }

    #[test]
    fn confidence_vector_replaces_negative_with_text_baseline() {
        use super::confidence_vector;
        let segments = vec![seg("x", Some(-0.5))];
        let vec = confidence_vector(&segments);
        let expected = 1.0_f64.ln_1p() + 1.0;
        assert!((vec[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn confidence_vector_replaces_nan_with_text_baseline() {
        use super::confidence_vector;
        let segments = vec![seg("abc", Some(f64::NAN))];
        let vec = confidence_vector(&segments);
        let expected = 3.0_f64.ln_1p() + 1.0;
        assert!(
            (vec[0] - expected).abs() < 1e-6,
            "NaN should be replaced with text baseline"
        );
    }

    #[test]
    fn confidence_vector_replaces_infinity_with_text_baseline() {
        use super::confidence_vector;
        let segments = vec![seg("abcd", Some(f64::INFINITY))];
        let vec = confidence_vector(&segments);
        let expected = 4.0_f64.ln_1p() + 1.0;
        assert!((vec[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn confidence_vector_empty_text_uses_max_1_chars() {
        use super::confidence_vector;
        let segments = vec![seg("", None)];
        let vec = confidence_vector(&segments);
        // "".chars().count() = 0, .max(1) = 1 → baseline = 1.0_f64.ln_1p() + 1.0
        let expected = 1.0_f64.ln_1p() + 1.0;
        assert!((vec[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn confidence_vector_unicode_text_counts_chars() {
        use super::confidence_vector;
        // "héllo" has 5 chars
        let segments = vec![seg("héllo", None)];
        let vec = confidence_vector(&segments);
        let expected = 5.0_f64.ln_1p() + 1.0;
        assert!((vec[0] - expected).abs() < 1e-6);
    }

    // --- Direct apply_confidences tests ---

    #[test]
    fn apply_confidences_sets_values() {
        use super::apply_confidences;
        let mut segments = vec![seg("a", None), seg("b", Some(0.5))];
        apply_confidences(&mut segments, &[0.3, 0.7]);
        assert_eq!(segments[0].confidence, Some(0.3));
        assert_eq!(segments[1].confidence, Some(0.7));
    }

    #[test]
    fn apply_confidences_with_fewer_values_leaves_excess_segments_unchanged() {
        use super::apply_confidences;
        let mut segments = vec![
            seg("a", Some(0.1)),
            seg("b", Some(0.2)),
            seg("c", Some(0.3)),
        ];
        // Only 2 values for 3 segments — zip stops at shorter.
        apply_confidences(&mut segments, &[0.9, 0.8]);
        assert_eq!(segments[0].confidence, Some(0.9));
        assert_eq!(segments[1].confidence, Some(0.8));
        assert_eq!(segments[2].confidence, Some(0.3)); // unchanged
    }

    #[test]
    fn apply_confidences_empty_segments() {
        use super::apply_confidences;
        let mut segments: Vec<TranscriptionSegment> = vec![];
        apply_confidences(&mut segments, &[0.5]);
        assert!(segments.is_empty());
    }

    #[test]
    fn apply_confidences_empty_values() {
        use super::apply_confidences;
        let mut segments = vec![seg("a", Some(0.5))];
        apply_confidences(&mut segments, &[]);
        // No values to apply — original confidence unchanged.
        assert_eq!(segments[0].confidence, Some(0.5));
    }

    #[test]
    fn apply_confidences_sanitizes_non_finite_values() {
        use super::apply_confidences;
        let mut segments = vec![
            seg("a", None),
            seg("b", None),
            seg("c", None),
            seg("d", None),
        ];
        // NaN/Inf must not panic the downstream confidence clamps and must be
        // sanitized to a finite value in [0, 1]; finite values keep their value.
        apply_confidences(
            &mut segments,
            &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.42],
        );
        assert_eq!(segments[0].confidence, Some(0.0));
        assert_eq!(segments[1].confidence, Some(0.0));
        assert_eq!(segments[2].confidence, Some(0.0));
        assert_eq!(segments[3].confidence, Some(0.42));
        for segment in &segments {
            let c = segment.confidence.expect("confidence set");
            assert!(c.is_finite(), "confidence must be finite, got {c}");
            assert!((0.0..=1.0).contains(&c), "confidence in [0,1], got {c}");
        }
    }

    #[cfg(feature = "gpu-frankentorch")]
    #[test]
    fn normalize_with_frankentorch_non_finite_falls_back_to_finite() {
        use super::normalize_with_frankentorch;
        // A NaN-carrying input must never yield a non-finite normalized vector;
        // the GPU path validates its output and falls back to normalize_cpu.
        let out = normalize_with_frankentorch(&[f64::NAN, 1.0, 2.0])
            .expect("normalize_with_frankentorch should not error");
        assert_eq!(out.len(), 3);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "normalized output must be finite, got {out:?}"
        );
    }

    #[cfg(feature = "gpu-frankenjax")]
    #[test]
    fn normalize_with_frankenjax_non_finite_stays_finite() {
        use super::normalize_with_frankenjax;
        // A NaN in the input makes the raw ReduceSumVec total non-finite; the
        // finiteness guard degrades to a uniform (finite) distribution.
        let out = normalize_with_frankenjax(&[f64::NAN, 1.0, 2.0])
            .expect("normalize_with_frankenjax should not error");
        assert_eq!(out.len(), 3);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "normalized output must be finite, got {out:?}"
        );
    }

    // --- Direct build_report tests ---

    #[test]
    fn build_report_populates_all_fields() {
        use super::build_report;
        let mut result = make_result(vec![seg("a", Some(0.4)), seg("b", Some(0.6))]);
        let values = vec![0.3, 0.7];
        let report = build_report(
            &mut result,
            AccelerationBackend::None,
            values,
            Some(1.0),
            vec!["test note".to_owned()],
        );
        assert_eq!(report.backend, AccelerationBackend::None);
        assert_eq!(report.input_values, 2);
        assert!(report.normalized_confidences);
        assert_eq!(report.pre_mass, Some(1.0));
        assert!((report.post_mass.unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(report.notes, vec!["test note"]);
        // Segments should have been updated.
        assert_eq!(result.segments[0].confidence, Some(0.3));
        assert_eq!(result.segments[1].confidence, Some(0.7));
    }

    // --- normalize_cpu edge cases ---

    #[test]
    fn normalize_cpu_single_value() {
        use super::normalize_cpu;
        let result = normalize_cpu(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_cpu_mixed_valid_and_invalid() {
        use super::normalize_cpu;
        let result = normalize_cpu(&[2.0, f64::NAN, 3.0]);
        assert_eq!(result.len(), 3);
        // NaN maps to 0.0 in output, valid values normalized.
        assert!((result[0] - 0.4).abs() < 1e-6, "2/(2+3) = 0.4");
        assert!((result[1] - 0.0).abs() < 1e-6, "NaN maps to 0.0");
        assert!((result[2] - 0.6).abs() < 1e-6, "3/(2+3) = 0.6");
    }

    #[test]
    fn normalize_cpu_all_negative_gives_uniform() {
        use super::normalize_cpu;
        let result = normalize_cpu(&[-1.0, -2.0, -3.0]);
        assert_eq!(result.len(), 3);
        // All negative → safe_sum = 0 → uniform 1/3
        for v in &result {
            assert!(
                (*v - 1.0 / 3.0).abs() < 1e-6,
                "all negative should give uniform 1/3, got {v}"
            );
        }
    }

    #[test]
    fn normalize_cpu_empty_does_not_panic() {
        use super::normalize_cpu;
        let result = normalize_cpu(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn apply_sets_acceleration_field_on_result() {
        let mut result = make_result(vec![seg("test", Some(0.5))]);
        assert!(result.acceleration.is_none());

        apply(&mut result);
        assert!(
            result.acceleration.is_some(),
            "apply should set acceleration field"
        );
    }

    #[test]
    fn apply_cpu_fallback_notes_contain_cpu_message() {
        let mut result = make_result(vec![seg("test", Some(0.5))]);
        let report = apply(&mut result);
        // Without GPU features, the CPU fallback note should be present
        assert!(
            report.notes.iter().any(|n| n.contains("CPU")),
            "notes should mention CPU fallback: {:?}",
            report.notes
        );
    }

    #[test]
    fn confidence_vector_long_text_produces_higher_baseline() {
        use super::confidence_vector;
        let short_seg = seg("hi", None);
        let long_seg = seg(&"a".repeat(1000), None);
        let short_vec = confidence_vector(&[short_seg]);
        let long_vec = confidence_vector(&[long_seg]);
        assert!(
            long_vec[0] > short_vec[0],
            "longer text should produce higher baseline"
        );
    }
    // -----------------------------------------------------------------------
    // Comprehensive accelerate module unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn cpu_normalization_all_zero_confidence() {
        // Segments with all-zero confidence should get uniform distribution.
        use super::normalize_cpu;

        let zeros = vec![0.0; 4];
        let result = normalize_cpu(&zeros);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(
                (*v - 0.25).abs() < 1e-9,
                "all-zero input should produce uniform 1/4, got {v}"
            );
        }

        // Also validate through the full pipeline: zero confidence triggers
        // text-weight baseline in confidence_vector, then normalizes to 1.0.
        let mut tr = make_result(vec![
            seg("aaa", Some(0.0)),
            seg("bbb", Some(0.0)),
            seg("ccc", Some(0.0)),
            seg("ddd", Some(0.0)),
        ]);
        let report = apply(&mut tr);
        assert!(report.normalized_confidences);
        let sum: f64 = tr.segments.iter().filter_map(|s| s.confidence).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "all-zero confidences through pipeline should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn cpu_normalization_single_segment() {
        // Exactly one segment should normalize to 1.0.
        use super::normalize_cpu;

        let result = normalize_cpu(&[42.0]);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 1.0).abs() < 1e-9,
            "single segment should normalize to 1.0, got {}",
            result[0]
        );

        // Through the full pipeline.
        let mut tr = make_result(vec![seg("only one", Some(3.21))]);
        let report = apply(&mut tr);
        assert_eq!(report.input_values, 1);
        let c = tr.segments[0].confidence.unwrap();
        assert!(
            (c - 1.0).abs() < 1e-9,
            "single segment via apply should be 1.0, got {c}"
        );
    }

    #[test]
    fn cpu_normalization_nan_confidence() {
        // NaN confidence values must be handled safely: no panics, finite output.
        use super::normalize_cpu;

        let result = normalize_cpu(&[f64::NAN, f64::NAN, f64::NAN]);
        assert_eq!(result.len(), 3);
        for v in &result {
            assert!(
                v.is_finite(),
                "NaN input should produce finite output, got {v}"
            );
        }
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "all-NaN should produce uniform distribution summing to 1.0, got {sum}"
        );

        // Mixed NaN and valid.
        let mixed = normalize_cpu(&[f64::NAN, 4.0, f64::NAN, 6.0]);
        assert_eq!(mixed.len(), 4);
        assert!((mixed[0] - 0.0).abs() < 1e-9, "NaN maps to 0.0");
        assert!(
            (mixed[1] - 0.4).abs() < 1e-9,
            "4/(4+6) = 0.4, got {}",
            mixed[1]
        );
        assert!((mixed[2] - 0.0).abs() < 1e-9, "NaN maps to 0.0");
        assert!(
            (mixed[3] - 0.6).abs() < 1e-9,
            "6/(4+6) = 0.6, got {}",
            mixed[3]
        );
    }

    #[test]
    fn cpu_normalization_inf_confidence() {
        // Infinity confidence values must be handled safely.
        use super::normalize_cpu;

        // All infinite → uniform distribution.
        let result = normalize_cpu(&[f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(result.len(), 2);
        for v in &result {
            assert!(
                v.is_finite(),
                "infinite input should produce finite output, got {v}"
            );
            assert!(
                (*v - 0.5).abs() < 1e-9,
                "all-infinite should give uniform 0.5, got {v}"
            );
        }

        // Positive infinity mixed with valid values.
        let mixed = normalize_cpu(&[f64::INFINITY, 3.0, 7.0]);
        assert_eq!(mixed.len(), 3);
        assert!(
            (mixed[0] - 0.0).abs() < 1e-9,
            "INFINITY should map to 0.0, got {}",
            mixed[0]
        );
        assert!(
            (mixed[1] - 0.3).abs() < 1e-9,
            "3/(3+7) = 0.3, got {}",
            mixed[1]
        );
        assert!(
            (mixed[2] - 0.7).abs() < 1e-9,
            "7/(3+7) = 0.7, got {}",
            mixed[2]
        );
    }

    #[test]
    fn cpu_normalization_negative_confidence() {
        // Negative confidence values should be treated as invalid (mapped to 0.0).
        use super::normalize_cpu;

        let result = normalize_cpu(&[-5.0, 3.0, -2.0, 7.0]);
        assert_eq!(result.len(), 4);

        // Negative values map to 0.0 in output.
        assert!(
            (result[0] - 0.0).abs() < 1e-9,
            "negative should map to 0.0, got {}",
            result[0]
        );
        assert!(
            (result[2] - 0.0).abs() < 1e-9,
            "negative should map to 0.0, got {}",
            result[2]
        );

        // Positive values normalized among themselves.
        assert!(
            (result[1] - 0.3).abs() < 1e-9,
            "3/(3+7) = 0.3, got {}",
            result[1]
        );
        assert!(
            (result[3] - 0.7).abs() < 1e-9,
            "7/(3+7) = 0.7, got {}",
            result[3]
        );

        // Through apply: negative confidence triggers text-weight baseline.
        let mut tr = make_result(vec![seg("neg", Some(-1.0)), seg("pos", Some(0.5))]);
        let report = apply(&mut tr);
        assert!(report.normalized_confidences);
        for s in &tr.segments {
            let c = s.confidence.unwrap();
            assert!(
                c >= 0.0,
                "output confidence should be non-negative, got {c}"
            );
            assert!(c.is_finite(), "output confidence should be finite, got {c}");
        }
    }

    #[test]
    fn cpu_normalization_very_large_segment_count() {
        // Test with 1000 segments to ensure no performance or precision issues.
        let segments: Vec<TranscriptionSegment> = (0..1000)
            .map(|i| seg(&format!("segment-{i}"), Some((i as f64 + 1.0) * 0.001)))
            .collect();
        let mut result = make_result(segments);

        let report = apply(&mut result);
        assert!(report.normalized_confidences);
        assert_eq!(report.input_values, 1000);
        assert_eq!(result.segments.len(), 1000);

        // All confidences should be finite and non-negative.
        for (i, s) in result.segments.iter().enumerate() {
            let c = s.confidence.unwrap();
            assert!(
                c.is_finite() && c >= 0.0,
                "segment {i}: confidence should be finite non-negative, got {c}"
            );
        }

        // Sum should be 1.0.
        let sum: f64 = result.segments.iter().filter_map(|s| s.confidence).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "1000 segments should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn cpu_normalization_preserves_relative_order() {
        // Higher input confidence must remain higher after normalization.
        use super::normalize_cpu;

        let values = vec![1.0, 5.0, 3.0, 10.0, 0.5];
        let normalized = normalize_cpu(&values);

        // Collect (original_index, normalized_value) and sort by normalized descending.
        let mut indexed: Vec<(usize, f64)> = normalized.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut original_indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        original_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // The rank order should be identical.
        let norm_order: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let orig_order: Vec<usize> = original_indexed.iter().map(|(i, _)| *i).collect();
        assert_eq!(
            norm_order, orig_order,
            "relative order should be preserved: original {orig_order:?}, normalized {norm_order:?}"
        );

        // Additionally, pairwise: if values[i] > values[j], then normalized[i] > normalized[j].
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                if values[i] > values[j] {
                    assert!(
                        normalized[i] > normalized[j],
                        "values[{i}]={} > values[{j}]={}, but normalized[{i}]={} <= normalized[{j}]={}",
                        values[i],
                        values[j],
                        normalized[i],
                        normalized[j]
                    );
                } else if values[i] < values[j] {
                    assert!(
                        normalized[i] < normalized[j],
                        "values[{i}]={} < values[{j}]={}, but normalized[{i}]={} >= normalized[{j}]={}",
                        values[i],
                        values[j],
                        normalized[i],
                        normalized[j]
                    );
                }
            }
        }
    }

    #[test]
    fn cpu_normalization_output_sums_to_one() {
        // Parametric test with various inputs — all must sum to 1.0.
        use super::normalize_cpu;

        let cases: Vec<(&str, Vec<f64>)> = vec![
            ("equal values", vec![1.0, 1.0, 1.0]),
            ("ascending", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("descending", vec![10.0, 5.0, 2.0, 1.0]),
            ("very small", vec![1e-10, 2e-10, 3e-10]),
            ("very large", vec![1e15, 2e15, 3e15]),
            ("mixed magnitude", vec![0.001, 1.0, 1000.0]),
            ("single element", vec![99.0]),
            ("two elements", vec![0.3, 0.7]),
            ("many equal", vec![1.0; 100]),
            ("powers of two", vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        ];

        for (label, values) in &cases {
            let normalized = normalize_cpu(values);
            assert_eq!(
                normalized.len(),
                values.len(),
                "{label}: output length mismatch"
            );
            let sum: f64 = normalized.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "{label}: expected sum 1.0, got {sum}"
            );
            for (i, v) in normalized.iter().enumerate() {
                assert!(
                    v.is_finite() && *v >= 0.0,
                    "{label}: normalized[{i}] should be finite non-negative, got {v}"
                );
            }
        }
    }

    #[test]
    fn confidence_vector_missing_confidence_uses_text_weight() {
        // When confidence is None, the deterministic baseline uses text weight.
        use super::confidence_vector;

        let segments = vec![seg("hello world", None), seg("x", None), seg("", None)];
        let vec = confidence_vector(&segments);
        assert_eq!(vec.len(), 3);

        // "hello world" = 11 chars → ln_1p(11) + 1.0
        let expected_0 = 11.0_f64.ln_1p() + 1.0;
        assert!(
            (vec[0] - expected_0).abs() < 1e-9,
            "expected {expected_0}, got {}",
            vec[0]
        );

        // "x" = 1 char → ln_1p(1) + 1.0
        let expected_1 = 1.0_f64.ln_1p() + 1.0;
        assert!(
            (vec[1] - expected_1).abs() < 1e-9,
            "expected {expected_1}, got {}",
            vec[1]
        );

        // "" = 0 chars, .max(1) = 1 → ln_1p(1) + 1.0
        let expected_2 = 1.0_f64.ln_1p() + 1.0;
        assert!(
            (vec[2] - expected_2).abs() < 1e-9,
            "expected {expected_2}, got {}",
            vec[2]
        );

        // Longer text should produce a larger baseline value.
        assert!(
            vec[0] > vec[1],
            "11-char text should have higher baseline than 1-char text"
        );
    }

    #[test]
    fn confidence_vector_all_present() {
        // When all segments have valid positive finite confidence, values pass through unchanged.
        use super::confidence_vector;

        let segments = vec![
            seg("alpha", Some(0.95)),
            seg("beta", Some(0.42)),
            seg("gamma", Some(0.73)),
            seg("delta", Some(0.11)),
        ];
        let vec = confidence_vector(&segments);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0], 0.95);
        assert_eq!(vec[1], 0.42);
        assert_eq!(vec[2], 0.73);
        assert_eq!(vec[3], 0.11);
    }

    #[test]
    fn apply_populates_acceleration_field() {
        // Verify result.acceleration is set after apply().
        let mut result = make_result(vec![seg("first", Some(0.3)), seg("second", Some(0.7))]);
        assert!(
            result.acceleration.is_none(),
            "acceleration should be None before apply"
        );

        let report = apply(&mut result);

        assert!(
            result.acceleration.is_some(),
            "acceleration should be populated after apply"
        );

        let accel = result.acceleration.as_ref().unwrap();
        assert_eq!(accel.backend, report.backend);
        assert_eq!(accel.input_values, report.input_values);
        assert_eq!(accel.normalized_confidences, report.normalized_confidences);
        assert_eq!(accel.pre_mass, report.pre_mass);
        assert_eq!(accel.post_mass, report.post_mass);
        assert_eq!(accel.notes, report.notes);
    }

    #[test]
    fn apply_report_notes_contain_cpu_fallback() {
        // Without GPU feature flags, the report should note CPU fallback usage.
        let mut result = make_result(vec![seg("test segment", Some(0.5))]);
        let report = apply(&mut result);

        let has_cpu_note = report
            .notes
            .iter()
            .any(|note| note.contains("CPU") || note.contains("cpu"));
        assert!(
            has_cpu_note,
            "notes should mention CPU fallback when no GPU backend is available: {:?}",
            report.notes
        );

        // Check the specific expected note text.
        let has_deterministic_note = report
            .notes
            .iter()
            .any(|note| note.contains("deterministic CPU normalization fallback"));
        assert!(
            has_deterministic_note,
            "notes should contain 'deterministic CPU normalization fallback': {:?}",
            report.notes
        );
    }

    #[test]
    fn apply_preserves_existing_segment_text() {
        // Normalization must not modify segment text or timestamps.
        let mut result = make_result(vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.5),
                text: "Hello, world!".to_owned(),
                speaker: Some("speaker_0".to_owned()),
                confidence: Some(0.8),
            },
            TranscriptionSegment {
                start_sec: Some(1.5),
                end_sec: Some(3.0),
                text: "How are you?".to_owned(),
                speaker: None,
                confidence: Some(0.6),
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "No timestamps".to_owned(),
                speaker: Some("speaker_1".to_owned()),
                confidence: None,
            },
        ]);

        apply(&mut result);

        // Text must be unchanged.
        assert_eq!(result.segments[0].text, "Hello, world!");
        assert_eq!(result.segments[1].text, "How are you?");
        assert_eq!(result.segments[2].text, "No timestamps");

        // Timestamps must be unchanged.
        assert_eq!(result.segments[0].start_sec, Some(0.0));
        assert_eq!(result.segments[0].end_sec, Some(1.5));
        assert_eq!(result.segments[1].start_sec, Some(1.5));
        assert_eq!(result.segments[1].end_sec, Some(3.0));
        assert_eq!(result.segments[2].start_sec, None);
        assert_eq!(result.segments[2].end_sec, None);

        // Speaker labels must be unchanged.
        assert_eq!(result.segments[0].speaker, Some("speaker_0".to_owned()));
        assert_eq!(result.segments[1].speaker, None);
        assert_eq!(result.segments[2].speaker, Some("speaker_1".to_owned()));

        // Confidences should now be set (modified by normalization), but should be valid.
        for s in &result.segments {
            let c = s.confidence.unwrap();
            assert!(c.is_finite() && c >= 0.0);
        }
    }

    // ===================================================================
    // bd-1r7.1: Attention layer acceleration tests
    // ===================================================================

    #[test]
    fn attention_cpu_self_attention_basic() {
        use super::{AttentionKind, attention_scores_cpu};

        let query = vec![1.0, 0.0, 1.0, 0.0];
        let key = vec![1.0, 1.0, 0.0, 0.0];
        let result = attention_scores_cpu(&query, &key, AttentionKind::SelfAttention);

        assert_eq!(result.kind, AttentionKind::SelfAttention);
        assert!(!result.gpu_accelerated);
        assert_eq!(result.scores.len(), 4);

        // Scores should sum to 1.0 (softmax output).
        let sum: f64 = result.scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "attention scores should sum to 1.0, got {sum}"
        );

        // All scores should be finite and non-negative.
        for (i, s) in result.scores.iter().enumerate() {
            assert!(
                s.is_finite() && *s >= 0.0,
                "score[{i}] should be finite non-negative, got {s}"
            );
        }
    }

    #[test]
    fn attention_cpu_cross_attention_basic() {
        use super::{AttentionKind, attention_scores_cpu};

        let query = vec![0.5, 0.5];
        let key = vec![1.0, 0.0];
        let result = attention_scores_cpu(&query, &key, AttentionKind::CrossAttention);

        assert_eq!(result.kind, AttentionKind::CrossAttention);
        assert!(!result.gpu_accelerated);
        assert_eq!(result.scores.len(), 2);

        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn attention_cpu_empty_inputs() {
        use super::{AttentionKind, attention_scores_cpu};

        let result = attention_scores_cpu(&[], &[], AttentionKind::SelfAttention);
        assert!(result.scores.is_empty());

        let result = attention_scores_cpu(&[1.0], &[], AttentionKind::SelfAttention);
        assert!(result.scores.is_empty());

        let result = attention_scores_cpu(&[], &[1.0], AttentionKind::CrossAttention);
        assert!(result.scores.is_empty());
    }

    #[test]
    fn attention_cpu_nan_and_inf_handling() {
        use super::{AttentionKind, attention_scores_cpu};

        let query = vec![f64::NAN, 1.0, f64::INFINITY];
        let key = vec![1.0, f64::NAN, 1.0];
        let result = attention_scores_cpu(&query, &key, AttentionKind::SelfAttention);

        assert_eq!(result.scores.len(), 3);
        for (i, s) in result.scores.iter().enumerate() {
            assert!(
                s.is_finite() && *s >= 0.0,
                "score[{i}] should be finite non-negative, got {s}"
            );
        }
        let sum: f64 = result.scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "attention scores should sum to 1.0 even with NaN/Inf inputs, got {sum}"
        );
    }

    #[test]
    fn attention_cpu_unequal_lengths_uses_minimum() {
        use super::{AttentionKind, attention_scores_cpu};

        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let key = vec![0.5, 0.5, 0.5];
        let result = attention_scores_cpu(&query, &key, AttentionKind::SelfAttention);

        // Should only produce scores for min(5, 3) = 3 elements.
        assert_eq!(result.scores.len(), 3);
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn attention_cpu_identical_qk_produces_uniform() {
        use super::{AttentionKind, attention_scores_cpu};

        // When Q and K are identical uniform vectors, attention should be close to uniform.
        let uniform = vec![1.0; 8];
        let result = attention_scores_cpu(&uniform, &uniform, AttentionKind::SelfAttention);

        assert_eq!(result.scores.len(), 8);
        for (i, s) in result.scores.iter().enumerate() {
            assert!(
                (*s - 0.125).abs() < 1e-6,
                "score[{i}] should be ~0.125 (uniform), got {s}"
            );
        }
    }

    #[test]
    fn attention_cpu_single_element() {
        use super::{AttentionKind, attention_scores_cpu};

        let result = attention_scores_cpu(&[3.0], &[2.0], AttentionKind::SelfAttention);
        assert_eq!(result.scores.len(), 1);
        assert!(
            (result.scores[0] - 1.0).abs() < 1e-9,
            "single-element attention should be 1.0"
        );
    }

    #[test]
    fn compute_attention_dispatch_uses_cpu_without_features() {
        use super::{AttentionKind, compute_attention};

        let result = compute_attention(&[1.0, 2.0], &[2.0, 1.0], AttentionKind::SelfAttention);

        // Without gpu-frankentorch feature, this must be CPU.
        #[cfg(not(feature = "gpu-frankentorch"))]
        assert!(!result.gpu_accelerated);

        assert_eq!(result.scores.len(), 2);
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // ===================================================================
    // bd-1r7.1: Embedding layer acceleration tests
    // ===================================================================

    #[test]
    fn embedding_cpu_basic_lookup() {
        use super::{EmbeddingKind, embedding_lookup_cpu};

        let table = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = embedding_lookup_cpu(&[0, 2, 1], &table, EmbeddingKind::Token);

        assert_eq!(result.kind, EmbeddingKind::Token);
        assert!(!result.gpu_accelerated);
        assert_eq!(result.embeddings.len(), 3);
        assert_eq!(result.embeddings[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(result.embeddings[1], vec![0.0, 0.0, 1.0]);
        assert_eq!(result.embeddings[2], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn embedding_cpu_positional() {
        use super::{EmbeddingKind, embedding_lookup_cpu};

        let table = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let result = embedding_lookup_cpu(&[0, 1], &table, EmbeddingKind::Positional);

        assert_eq!(result.kind, EmbeddingKind::Positional);
        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(result.embeddings[0], vec![0.1, 0.2]);
        assert_eq!(result.embeddings[1], vec![0.3, 0.4]);
    }

    #[test]
    fn embedding_cpu_out_of_range_index_returns_zero_vector() {
        use super::{EmbeddingKind, embedding_lookup_cpu};

        let table = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = embedding_lookup_cpu(&[0, 99, 1], &table, EmbeddingKind::Token);

        assert_eq!(result.embeddings.len(), 3);
        assert_eq!(result.embeddings[0], vec![1.0, 2.0]);
        assert_eq!(result.embeddings[1], vec![0.0, 0.0]); // out of range
        assert_eq!(result.embeddings[2], vec![3.0, 4.0]);
    }

    #[test]
    fn embedding_cpu_empty_inputs() {
        use super::{EmbeddingKind, embedding_lookup_cpu};

        // Empty indices.
        let table = vec![vec![1.0]];
        let result = embedding_lookup_cpu(&[], &table, EmbeddingKind::Token);
        assert!(result.embeddings.is_empty());

        // Empty table.
        let result = embedding_lookup_cpu(&[0], &[], EmbeddingKind::Token);
        assert_eq!(result.embeddings.len(), 1);
        assert!(result.embeddings[0].is_empty()); // embed_dim = 0
    }

    #[test]
    fn embedding_cpu_nan_in_table_sanitized() {
        use super::{EmbeddingKind, embedding_lookup_cpu};

        let table = vec![vec![f64::NAN, 1.0, f64::INFINITY]];
        let result = embedding_lookup_cpu(&[0], &table, EmbeddingKind::Token);

        assert_eq!(result.embeddings.len(), 1);
        assert_eq!(result.embeddings[0], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn embedding_cpu_repeated_indices() {
        use super::{EmbeddingKind, embedding_lookup_cpu};

        let table = vec![vec![10.0], vec![20.0]];
        let result = embedding_lookup_cpu(&[0, 0, 1, 1, 0], &table, EmbeddingKind::Token);

        assert_eq!(result.embeddings.len(), 5);
        assert_eq!(result.embeddings[0], vec![10.0]);
        assert_eq!(result.embeddings[1], vec![10.0]);
        assert_eq!(result.embeddings[2], vec![20.0]);
        assert_eq!(result.embeddings[3], vec![20.0]);
        assert_eq!(result.embeddings[4], vec![10.0]);
    }

    #[test]
    fn compute_embedding_dispatch_uses_cpu_without_features() {
        use super::{EmbeddingKind, compute_embedding};

        let table = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = compute_embedding(&[1, 0], &table, EmbeddingKind::Token);

        #[cfg(not(feature = "gpu-frankentorch"))]
        assert!(!result.gpu_accelerated);

        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(result.embeddings[0], vec![3.0, 4.0]);
        assert_eq!(result.embeddings[1], vec![1.0, 2.0]);
    }

    // ===================================================================
    // bd-1r7.1: VAD scoring acceleration tests
    // ===================================================================

    #[test]
    fn vad_cpu_basic_scoring() {
        use super::vad_scores_cpu;

        let energies = vec![0.0, 5.0, -5.0, 10.0];
        let result = vad_scores_cpu(&energies);

        assert!(!result.gpu_accelerated);
        assert_eq!(result.frame_scores.len(), 4);

        // sigmoid(0) = 0.5
        assert!(
            (result.frame_scores[0] - 0.5).abs() < 1e-9,
            "sigmoid(0) should be 0.5, got {}",
            result.frame_scores[0]
        );

        // sigmoid(5) should be close to 1.0
        assert!(
            result.frame_scores[1] > 0.99,
            "sigmoid(5) should be >0.99, got {}",
            result.frame_scores[1]
        );

        // sigmoid(-5) should be close to 0.0
        assert!(
            result.frame_scores[2] < 0.01,
            "sigmoid(-5) should be <0.01, got {}",
            result.frame_scores[2]
        );

        // sigmoid(10) should be very close to 1.0
        assert!(
            result.frame_scores[3] > 0.999,
            "sigmoid(10) should be >0.999, got {}",
            result.frame_scores[3]
        );

        // Activity ratio should be the mean of frame scores.
        let expected_ratio = result.frame_scores.iter().sum::<f64>() / 4.0;
        assert!(
            (result.activity_ratio - expected_ratio).abs() < 1e-9,
            "activity_ratio should be mean of scores"
        );
    }

    #[test]
    fn vad_cpu_empty_input() {
        use super::vad_scores_cpu;

        let result = vad_scores_cpu(&[]);
        assert!(result.frame_scores.is_empty());
        assert!((result.activity_ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn vad_cpu_nan_and_inf_handling() {
        use super::vad_scores_cpu;

        let energies = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0];
        let result = vad_scores_cpu(&energies);

        assert_eq!(result.frame_scores.len(), 4);
        // NaN/Inf should map to 0.0.
        assert!(
            (result.frame_scores[0] - 0.0).abs() < 1e-9,
            "NaN energy should give score 0.0"
        );
        assert!(
            (result.frame_scores[1] - 0.0).abs() < 1e-9,
            "Inf energy should give score 0.0"
        );
        assert!(
            (result.frame_scores[2] - 0.0).abs() < 1e-9,
            "-Inf energy should give score 0.0"
        );
        // sigmoid(0) = 0.5
        assert!((result.frame_scores[3] - 0.5).abs() < 1e-9);

        // All scores should be in [0, 1].
        for (i, s) in result.frame_scores.iter().enumerate() {
            assert!(
                *s >= 0.0 && *s <= 1.0,
                "frame_scores[{i}] should be in [0,1], got {s}"
            );
        }
    }

    #[test]
    fn vad_cpu_all_high_energy_gives_high_ratio() {
        use super::vad_scores_cpu;

        let energies = vec![100.0; 10];
        let result = vad_scores_cpu(&energies);

        assert!(
            result.activity_ratio > 0.99,
            "all high-energy frames should give ratio near 1.0, got {}",
            result.activity_ratio
        );
    }

    #[test]
    fn vad_cpu_all_low_energy_gives_low_ratio() {
        use super::vad_scores_cpu;

        let energies = vec![-100.0; 10];
        let result = vad_scores_cpu(&energies);

        assert!(
            result.activity_ratio < 0.01,
            "all low-energy frames should give ratio near 0.0, got {}",
            result.activity_ratio
        );
    }

    #[test]
    fn vad_cpu_single_frame() {
        use super::vad_scores_cpu;

        let result = vad_scores_cpu(&[0.0]);
        assert_eq!(result.frame_scores.len(), 1);
        assert!((result.frame_scores[0] - 0.5).abs() < 1e-9);
        assert!((result.activity_ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn compute_vad_scores_dispatch_uses_cpu_without_features() {
        use super::compute_vad_scores;

        let result = compute_vad_scores(&[0.0, 5.0, -5.0]);

        #[cfg(not(feature = "gpu-frankentorch"))]
        assert!(!result.gpu_accelerated);

        assert_eq!(result.frame_scores.len(), 3);
        for s in &result.frame_scores {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    // ===================================================================
    // bd-1r7.1: Shared math utility tests (softmax, sigmoid)
    // ===================================================================

    #[test]
    fn softmax_cpu_basic() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);

        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // Higher input should produce higher output.
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn softmax_cpu_empty() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn softmax_cpu_single_element() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[42.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_cpu_equal_values_uniform() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(
                (*v - 0.25).abs() < 1e-9,
                "equal inputs should produce uniform softmax, got {v}"
            );
        }
    }

    #[test]
    fn softmax_cpu_nan_handling() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[f64::NAN, 1.0, 2.0]);
        assert_eq!(result.len(), 3);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for v in &result {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[test]
    fn softmax_cpu_large_values_numerically_stable() {
        use super::softmax_cpu;

        // Large values that would overflow naive exp() without max subtraction.
        let result = softmax_cpu(&[1000.0, 1001.0, 1002.0]);
        assert_eq!(result.len(), 3);
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "softmax should sum to 1.0 even with large inputs, got {sum}"
        );
        for v in &result {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[test]
    fn softmax_cpu_negative_values() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[-1.0, -2.0, -3.0]);
        assert_eq!(result.len(), 3);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Higher (less negative) should still produce higher softmax output.
        assert!(result[0] > result[1]);
        assert!(result[1] > result[2]);
    }

    #[test]
    fn sigmoid_basic_values() {
        use super::sigmoid;

        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
        assert!((sigmoid(f64::NAN) - 0.0).abs() < 1e-9);
        assert!((sigmoid(f64::INFINITY) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn sigmoid_monotonicity() {
        use super::sigmoid;

        let values: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        for window in values.windows(2) {
            let (a, b) = (window[0], window[1]);
            assert!(
                sigmoid(b) >= sigmoid(a),
                "sigmoid should be monotonically increasing: sigmoid({a}) = {}, sigmoid({b}) = {}",
                sigmoid(a),
                sigmoid(b)
            );
        }
    }

    // ===================================================================
    // bd-1r7.2: JIT-compiled inference tests
    // ===================================================================

    #[test]
    fn jit_kernel_cache_new_is_empty() {
        use super::JitKernelCache;

        let cache = JitKernelCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn jit_kernel_cache_default_is_empty() {
        use super::JitKernelCache;

        let cache = JitKernelCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn jit_kernel_cache_insert_and_get() {
        use super::{CompiledKernel, InferenceGraphPattern, JitKernelCache};

        let mut cache = JitKernelCache::new();
        assert!(
            cache
                .get(&InferenceGraphPattern::NormalizeSoftmax)
                .is_none()
        );

        let kernel = CompiledKernel {
            pattern: InferenceGraphPattern::NormalizeSoftmax,
            kernel_id: "test_kernel_v1".to_owned(),
        };
        cache.insert(kernel);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let retrieved = cache.get(&InferenceGraphPattern::NormalizeSoftmax);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().kernel_id, "test_kernel_v1");
    }

    #[test]
    fn jit_kernel_cache_multiple_patterns() {
        use super::{CompiledKernel, InferenceGraphPattern, JitKernelCache};

        let mut cache = JitKernelCache::new();

        cache.insert(CompiledKernel {
            pattern: InferenceGraphPattern::NormalizeSoftmax,
            kernel_id: "ns_v1".to_owned(),
        });
        cache.insert(CompiledKernel {
            pattern: InferenceGraphPattern::LinearActivation,
            kernel_id: "la_v1".to_owned(),
        });
        cache.insert(CompiledKernel {
            pattern: InferenceGraphPattern::AttentionBlock,
            kernel_id: "ab_v1".to_owned(),
        });

        assert_eq!(cache.len(), 3);
        assert!(
            cache
                .get(&InferenceGraphPattern::NormalizeSoftmax)
                .is_some()
        );
        assert!(
            cache
                .get(&InferenceGraphPattern::LinearActivation)
                .is_some()
        );
        assert!(cache.get(&InferenceGraphPattern::AttentionBlock).is_some());
    }

    #[test]
    fn jit_kernel_cache_overwrite_same_pattern() {
        use super::{CompiledKernel, InferenceGraphPattern, JitKernelCache};

        let mut cache = JitKernelCache::new();

        cache.insert(CompiledKernel {
            pattern: InferenceGraphPattern::NormalizeSoftmax,
            kernel_id: "v1".to_owned(),
        });
        cache.insert(CompiledKernel {
            pattern: InferenceGraphPattern::NormalizeSoftmax,
            kernel_id: "v2".to_owned(),
        });

        assert_eq!(cache.len(), 1);
        assert_eq!(
            cache
                .get(&InferenceGraphPattern::NormalizeSoftmax)
                .unwrap()
                .kernel_id,
            "v2"
        );
    }

    #[test]
    fn jit_kernel_cache_clear() {
        use super::{CompiledKernel, InferenceGraphPattern, JitKernelCache};

        let mut cache = JitKernelCache::new();
        cache.insert(CompiledKernel {
            pattern: InferenceGraphPattern::NormalizeSoftmax,
            kernel_id: "k1".to_owned(),
        });
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(
            cache
                .get(&InferenceGraphPattern::NormalizeSoftmax)
                .is_none()
        );
    }

    #[test]
    fn jit_normalize_softmax_cpu_basic() {
        use super::jit_normalize_softmax_cpu;

        let result = jit_normalize_softmax_cpu(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);

        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "normalize-then-softmax should sum to 1.0, got {sum}"
        );

        for v in &result {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[test]
    fn jit_normalize_softmax_cpu_empty() {
        use super::jit_normalize_softmax_cpu;

        let result = jit_normalize_softmax_cpu(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn jit_normalize_softmax_cpu_single() {
        use super::jit_normalize_softmax_cpu;

        let result = jit_normalize_softmax_cpu(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jit_linear_activation_cpu_basic() {
        use super::jit_linear_activation_cpu;

        let inputs = vec![1.0, -1.0, 2.0];
        let weights = vec![2.0, 3.0, 0.5];
        let result = jit_linear_activation_cpu(&inputs, &weights, 0.0);

        assert_eq!(result.len(), 3);
        // 1.0 * 2.0 + 0.0 = 2.0, relu(2.0) = 2.0
        assert!((result[0] - 2.0).abs() < 1e-9);
        // -1.0 * 3.0 + 0.0 = -3.0, relu(-3.0) = 0.0
        assert!((result[1] - 0.0).abs() < 1e-9);
        // 2.0 * 0.5 + 0.0 = 1.0, relu(1.0) = 1.0
        assert!((result[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jit_linear_activation_cpu_with_bias() {
        use super::jit_linear_activation_cpu;

        let inputs = vec![1.0, -1.0];
        let weights = vec![1.0, 1.0];
        let result = jit_linear_activation_cpu(&inputs, &weights, 0.5);

        // 1.0 * 1.0 + 0.5 = 1.5, relu(1.5) = 1.5
        assert!((result[0] - 1.5).abs() < 1e-9);
        // -1.0 * 1.0 + 0.5 = -0.5, relu(-0.5) = 0.0
        assert!((result[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn jit_linear_activation_cpu_nan_handling() {
        use super::jit_linear_activation_cpu;

        let result = jit_linear_activation_cpu(&[f64::NAN, 1.0], &[1.0, f64::NAN], 0.0);
        assert_eq!(result.len(), 2);
        // NaN * 1.0 → 0.0 * 1.0 = 0.0, relu(0.0) = 0.0
        assert!((result[0] - 0.0).abs() < 1e-9);
        // 1.0 * NaN → 1.0 * 0.0 = 0.0, relu(0.0) = 0.0
        assert!((result[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn jit_linear_activation_cpu_empty() {
        use super::jit_linear_activation_cpu;

        let result = jit_linear_activation_cpu(&[], &[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn jit_attention_block_cpu_basic() {
        use super::jit_attention_block_cpu;

        let query = vec![1.0, 0.0, 1.0];
        let key = vec![1.0, 1.0, 0.0];
        let value = vec![10.0, 20.0, 30.0];
        let result = jit_attention_block_cpu(&query, &key, &value);

        assert_eq!(result.len(), 3);
        // All output values should be finite and non-negative (since weights
        // from softmax are non-negative and values are positive).
        for (i, v) in result.iter().enumerate() {
            assert!(
                v.is_finite() && *v >= 0.0,
                "output[{i}] should be finite non-negative, got {v}"
            );
        }
    }

    #[test]
    fn jit_attention_block_cpu_empty_inputs() {
        use super::jit_attention_block_cpu;

        assert!(jit_attention_block_cpu(&[], &[1.0], &[1.0]).is_empty());
        assert!(jit_attention_block_cpu(&[1.0], &[], &[1.0]).is_empty());
        assert!(jit_attention_block_cpu(&[1.0], &[1.0], &[]).is_empty());
    }

    #[test]
    fn jit_attention_block_cpu_nan_handling() {
        use super::jit_attention_block_cpu;

        let result = jit_attention_block_cpu(&[f64::NAN, 1.0], &[1.0, f64::NAN], &[5.0, 10.0]);
        assert_eq!(result.len(), 2);
        for v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn jit_inference_normalize_softmax_dispatch() {
        use super::{InferenceGraphPattern, JitKernelCache, jit_inference};

        let mut cache = JitKernelCache::new();
        let data = vec![3.0, 1.0, 2.0];

        let result = jit_inference(
            &InferenceGraphPattern::NormalizeSoftmax,
            &[data.as_slice()],
            &mut cache,
        );

        assert_eq!(result.pattern, InferenceGraphPattern::NormalizeSoftmax);
        assert!(!result.cache_hit, "first call should be a cache miss");
        assert_eq!(result.values.len(), 3);

        #[cfg(not(feature = "gpu-frankenjax"))]
        assert!(!result.gpu_accelerated);

        let sum: f64 = result.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // Second call should be a cache hit.
        let result2 = jit_inference(
            &InferenceGraphPattern::NormalizeSoftmax,
            &[data.as_slice()],
            &mut cache,
        );
        assert!(result2.cache_hit, "second call should be a cache hit");
    }

    #[test]
    fn jit_inference_linear_activation_dispatch() {
        use super::{InferenceGraphPattern, JitKernelCache, jit_inference};

        let mut cache = JitKernelCache::new();
        let data = vec![1.0, -1.0, 2.0];
        let weights = vec![2.0, 3.0, 0.5];

        let result = jit_inference(
            &InferenceGraphPattern::LinearActivation,
            &[data.as_slice(), weights.as_slice()],
            &mut cache,
        );

        assert_eq!(result.pattern, InferenceGraphPattern::LinearActivation);
        assert_eq!(result.values.len(), 3);

        // ReLU should zero out negative linear output.
        assert!(result.values[0] >= 0.0);
        assert!((result.values[1] - 0.0).abs() < 1e-9);
        assert!(result.values[2] >= 0.0);
    }

    #[test]
    fn jit_inference_attention_block_dispatch() {
        use super::{InferenceGraphPattern, JitKernelCache, jit_inference};

        let mut cache = JitKernelCache::new();
        let q = vec![1.0, 0.5];
        let k = vec![0.5, 1.0];
        let v = vec![10.0, 20.0];

        let result = jit_inference(
            &InferenceGraphPattern::AttentionBlock,
            &[q.as_slice(), k.as_slice(), v.as_slice()],
            &mut cache,
        );

        assert_eq!(result.pattern, InferenceGraphPattern::AttentionBlock);
        assert_eq!(result.values.len(), 2);
        for val in &result.values {
            assert!(val.is_finite() && *val >= 0.0);
        }
    }

    #[test]
    fn jit_inference_caching_across_patterns() {
        use super::{InferenceGraphPattern, JitKernelCache, jit_inference};

        let mut cache = JitKernelCache::new();
        let data = vec![1.0, 2.0];

        // Call NormalizeSoftmax.
        let r1 = jit_inference(
            &InferenceGraphPattern::NormalizeSoftmax,
            &[data.as_slice()],
            &mut cache,
        );
        assert!(!r1.cache_hit);

        // Call LinearActivation.
        let r2 = jit_inference(
            &InferenceGraphPattern::LinearActivation,
            &[data.as_slice(), data.as_slice()],
            &mut cache,
        );
        assert!(!r2.cache_hit);

        // Call NormalizeSoftmax again — should be cached.
        let r3 = jit_inference(
            &InferenceGraphPattern::NormalizeSoftmax,
            &[data.as_slice()],
            &mut cache,
        );
        assert!(r3.cache_hit);

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn jit_inference_empty_inputs_do_not_panic() {
        use super::{InferenceGraphPattern, JitKernelCache, jit_inference};

        let mut cache = JitKernelCache::new();

        let r1 = jit_inference(&InferenceGraphPattern::NormalizeSoftmax, &[&[]], &mut cache);
        assert!(r1.values.is_empty());

        let r2 = jit_inference(
            &InferenceGraphPattern::LinearActivation,
            &[&[], &[]],
            &mut cache,
        );
        assert!(r2.values.is_empty());

        let r3 = jit_inference(
            &InferenceGraphPattern::AttentionBlock,
            &[&[], &[], &[]],
            &mut cache,
        );
        assert!(r3.values.is_empty());
    }

    #[test]
    fn jit_inference_missing_inputs_do_not_panic() {
        use super::{InferenceGraphPattern, JitKernelCache, jit_inference};

        let mut cache = JitKernelCache::new();

        // No inputs at all for NormalizeSoftmax.
        let r1 = jit_inference(&InferenceGraphPattern::NormalizeSoftmax, &[], &mut cache);
        assert!(r1.values.is_empty());

        // No inputs for LinearActivation.
        let r2 = jit_inference(&InferenceGraphPattern::LinearActivation, &[], &mut cache);
        assert!(r2.values.is_empty());

        // No inputs for AttentionBlock.
        let r3 = jit_inference(&InferenceGraphPattern::AttentionBlock, &[], &mut cache);
        assert!(r3.values.is_empty());
    }

    #[test]
    fn inference_graph_pattern_equality() {
        use super::InferenceGraphPattern;

        assert_eq!(
            InferenceGraphPattern::NormalizeSoftmax,
            InferenceGraphPattern::NormalizeSoftmax
        );
        assert_ne!(
            InferenceGraphPattern::NormalizeSoftmax,
            InferenceGraphPattern::LinearActivation
        );
        assert_ne!(
            InferenceGraphPattern::LinearActivation,
            InferenceGraphPattern::AttentionBlock
        );
    }

    #[test]
    fn inference_graph_pattern_hash_consistency() {
        use super::InferenceGraphPattern;
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(InferenceGraphPattern::NormalizeSoftmax, "ns");
        map.insert(InferenceGraphPattern::LinearActivation, "la");
        map.insert(InferenceGraphPattern::AttentionBlock, "ab");

        assert_eq!(map.len(), 3);
        assert_eq!(map[&InferenceGraphPattern::NormalizeSoftmax], "ns");
        assert_eq!(map[&InferenceGraphPattern::LinearActivation], "la");
        assert_eq!(map[&InferenceGraphPattern::AttentionBlock], "ab");
    }

    #[test]
    fn compiled_kernel_clone() {
        use super::{CompiledKernel, InferenceGraphPattern};

        let kernel = CompiledKernel {
            pattern: InferenceGraphPattern::AttentionBlock,
            kernel_id: "test_kernel".to_owned(),
        };
        let cloned = kernel.clone();
        assert_eq!(cloned.pattern, InferenceGraphPattern::AttentionBlock);
        assert_eq!(cloned.kernel_id, "test_kernel");
    }

    #[test]
    fn jit_inference_result_fields() {
        use super::{InferenceGraphPattern, JitInferenceResult};

        let result = JitInferenceResult {
            values: vec![0.5, 0.5],
            pattern: InferenceGraphPattern::NormalizeSoftmax,
            cache_hit: false,
            gpu_accelerated: false,
        };

        assert_eq!(result.values.len(), 2);
        assert_eq!(result.pattern, InferenceGraphPattern::NormalizeSoftmax);
        assert!(!result.cache_hit);
        assert!(!result.gpu_accelerated);
    }

    #[test]
    fn jit_normalize_softmax_cpu_all_equal_produces_uniform() {
        use super::jit_normalize_softmax_cpu;

        let result = jit_normalize_softmax_cpu(&[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(
                (*v - 0.25).abs() < 1e-6,
                "equal inputs should produce uniform softmax, got {v}"
            );
        }
    }

    #[test]
    fn jit_linear_activation_cpu_all_relu_zero() {
        use super::jit_linear_activation_cpu;

        // All negative linear outputs should produce all zeros after ReLU.
        let result = jit_linear_activation_cpu(&[-1.0, -2.0, -3.0], &[1.0, 1.0, 1.0], 0.0);
        for v in &result {
            assert!((*v - 0.0).abs() < 1e-9, "should be 0.0 after ReLU, got {v}");
        }
    }

    #[test]
    fn jit_attention_block_cpu_preserves_value_information() {
        use super::jit_attention_block_cpu;

        // With uniform attention (identical Q and K), output should
        // approximate (1/dim * value[i]) for each position, since each
        // attention weight will be ~1/dim.
        let dim = 4;
        let uniform = vec![1.0; dim];
        let value = vec![4.0, 8.0, 12.0, 16.0];
        let result = jit_attention_block_cpu(&uniform, &uniform, &value);

        assert_eq!(result.len(), dim);
        // Each weight should be approximately 1/4 = 0.25.
        // So output[i] ~ 0.25 * value[i].
        for (i, r) in result.iter().enumerate() {
            let expected = 0.25 * value[i];
            assert!(
                (*r - expected).abs() < 1e-6,
                "output[{i}] should be ~{expected}, got {r}"
            );
        }
    }

    #[test]
    fn softmax_cpu_all_nan_values_returns_uniform() {
        use super::softmax_cpu;

        let result = softmax_cpu(&[f64::NAN, f64::NAN, f64::NAN]);
        assert_eq!(result.len(), 3);
        let expected = 1.0 / 3.0;
        for (i, v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-9,
                "element [{i}] should be ~{expected}, got {v}"
            );
            assert!(v.is_finite(), "element [{i}] should be finite");
        }

        // Also test with all INFINITY.
        let inf_result = softmax_cpu(&[f64::INFINITY, f64::INFINITY]);
        assert_eq!(inf_result.len(), 2);
        for v in &inf_result {
            assert!((v - 0.5).abs() < 1e-9, "should be uniform 0.5, got {v}");
        }
    }

    #[test]
    fn sigmoid_at_clamp_boundaries() {
        use super::sigmoid;

        // Values at and beyond the ±500 clamp boundaries.
        let at_neg500 = sigmoid(-500.0);
        let beyond_neg = sigmoid(-600.0);
        assert!(
            (at_neg500 - beyond_neg).abs() < 1e-15,
            "sigmoid(-500) and sigmoid(-600) should be equal due to clamping"
        );
        assert!(at_neg500 < 1e-200, "sigmoid(-500) should be near zero");

        let at_pos500 = sigmoid(500.0);
        let beyond_pos = sigmoid(600.0);
        assert!(
            (at_pos500 - beyond_pos).abs() < 1e-15,
            "sigmoid(500) and sigmoid(600) should be equal due to clamping"
        );
        assert!(
            (at_pos500 - 1.0).abs() < 1e-200,
            "sigmoid(500) should be near 1.0"
        );

        // Non-finite inputs return 0.0.
        assert_eq!(sigmoid(f64::NAN), 0.0);
        assert_eq!(sigmoid(f64::INFINITY), 0.0);
        assert_eq!(sigmoid(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn normalize_cpu_all_zero_values_returns_uniform() {
        use super::normalize_cpu;

        let result = normalize_cpu(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(
                (v - 0.25).abs() < 1e-9,
                "all-zero input should produce uniform 0.25, got {v}"
            );
        }

        // Also test with negative values (filtered out, same as zero-sum).
        let neg_result = normalize_cpu(&[-1.0, -2.0, -3.0]);
        assert_eq!(neg_result.len(), 3);
        let expected = 1.0 / 3.0;
        for v in &neg_result {
            assert!(
                (v - expected).abs() < 1e-9,
                "all-negative input should produce uniform {expected}, got {v}"
            );
        }
    }

    #[test]
    fn confidence_vector_with_zero_and_negative_confidence_uses_text_fallback() {
        use super::confidence_vector;

        let segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "hello".to_owned(),
                speaker: None,
                confidence: Some(0.0), // Zero → text-based fallback
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "world".to_owned(),
                speaker: None,
                confidence: Some(-0.5), // Negative → text-based fallback
            },
            TranscriptionSegment {
                start_sec: Some(2.0),
                end_sec: Some(3.0),
                text: "ok".to_owned(),
                speaker: None,
                confidence: Some(f64::NAN), // NaN → text-based fallback
            },
        ];

        let vector = confidence_vector(&segments);
        assert_eq!(vector.len(), 3);
        // All three should use the text-based formula: text.chars().count().max(1).ln_1p() + 1.0
        for (i, v) in vector.iter().enumerate() {
            assert!(v.is_finite(), "vector[{i}] should be finite: {v}");
            assert!(*v > 1.0, "text-based fallback should be > 1.0: {v}");
        }
        // "hello" (5 chars) should produce a larger value than "ok" (2 chars).
        assert!(
            vector[0] > vector[2],
            "5-char text should have higher baseline than 2-char: {} vs {}",
            vector[0],
            vector[2]
        );
    }

    #[test]
    fn normalize_cpu_mixed_finite_and_nan_ignores_nan() {
        use super::normalize_cpu;

        let result = normalize_cpu(&[2.0, f64::NAN, 3.0, f64::INFINITY]);
        assert_eq!(result.len(), 4);
        // safe_sum = 2.0 + 3.0 = 5.0 (NaN and INFINITY filtered out).
        assert!(
            (result[0] - 0.4).abs() < 1e-9,
            "2.0/5.0 = 0.4, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.0).abs() < 1e-9,
            "NaN maps to 0.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - 0.6).abs() < 1e-9,
            "3.0/5.0 = 0.6, got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.0).abs() < 1e-9,
            "INFINITY maps to 0.0, got {}",
            result[3]
        );
    }

    // ===================================================================
    // bd-1r7.1: Layer normalization tests
    // ===================================================================

    #[test]
    fn layer_norm_cpu_basic() {
        use super::layer_norm_cpu;

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let result = layer_norm_cpu(&values, &gamma, &beta, 1e-5);

        assert!(!result.gpu_accelerated);
        assert_eq!(result.normalized.len(), 5);

        // After layer norm with gamma=1, beta=0, the output should have
        // approximately zero mean.
        let mean: f64 = result.normalized.iter().sum::<f64>() / 5.0;
        assert!(
            mean.abs() < 1e-9,
            "layer norm output should have zero mean, got {mean}"
        );

        // All values should be finite.
        for (i, v) in result.normalized.iter().enumerate() {
            assert!(v.is_finite(), "normalized[{i}] should be finite, got {v}");
        }
    }

    #[test]
    fn layer_norm_cpu_with_gamma_and_beta() {
        use super::layer_norm_cpu;

        let values = vec![2.0, 4.0];
        let gamma = vec![2.0, 2.0];
        let beta = vec![1.0, 1.0];
        let result = layer_norm_cpu(&values, &gamma, &beta, 1e-5);

        assert_eq!(result.normalized.len(), 2);
        // Mean = 3.0, variance = 1.0, std = 1.0 (approx)
        // normed[0] = (2.0 - 3.0) / sqrt(1.0 + 1e-5) ≈ -1.0
        // output[0] = -1.0 * 2.0 + 1.0 = -1.0
        // normed[1] = (4.0 - 3.0) / sqrt(1.0 + 1e-5) ≈ 1.0
        // output[1] = 1.0 * 2.0 + 1.0 = 3.0
        assert!(
            (result.normalized[0] - (-1.0)).abs() < 1e-3,
            "expected ~-1.0, got {}",
            result.normalized[0]
        );
        assert!(
            (result.normalized[1] - 3.0).abs() < 1e-3,
            "expected ~3.0, got {}",
            result.normalized[1]
        );
    }

    #[test]
    fn layer_norm_cpu_empty_input() {
        use super::layer_norm_cpu;

        let result = layer_norm_cpu(&[], &[], &[], 1e-5);
        assert!(result.normalized.is_empty());
        assert!(!result.gpu_accelerated);
    }

    #[test]
    fn layer_norm_cpu_single_value() {
        use super::layer_norm_cpu;

        let result = layer_norm_cpu(&[5.0], &[1.0], &[0.0], 1e-5);
        assert_eq!(result.normalized.len(), 1);
        // Single value: mean = 5.0, variance = 0.0, std = sqrt(epsilon)
        // normed = (5.0 - 5.0) / sqrt(epsilon) = 0.0
        assert!(
            result.normalized[0].abs() < 1e-3,
            "single value layer norm should produce ~0.0, got {}",
            result.normalized[0]
        );
    }

    #[test]
    fn layer_norm_cpu_nan_handling() {
        use super::layer_norm_cpu;

        let values = vec![f64::NAN, 1.0, f64::INFINITY, 3.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let result = layer_norm_cpu(&values, &gamma, &beta, 1e-5);

        assert_eq!(result.normalized.len(), 4);
        for (i, v) in result.normalized.iter().enumerate() {
            assert!(v.is_finite(), "normalized[{i}] should be finite, got {v}");
        }
    }

    #[test]
    fn layer_norm_cpu_short_gamma_beta_uses_defaults() {
        use super::layer_norm_cpu;

        // gamma/beta shorter than values: missing positions use gamma=1.0, beta=0.0.
        let values = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0]; // only 1 element, positions 1,2 default to 1.0
        let beta = vec![10.0]; // only 1 element, positions 1,2 default to 0.0
        let result = layer_norm_cpu(&values, &gamma, &beta, 1e-5);

        assert_eq!(result.normalized.len(), 3);
        for v in &result.normalized {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn layer_norm_cpu_all_same_values() {
        use super::layer_norm_cpu;

        // All same values: mean = value, variance = 0, normed = 0.
        let values = vec![7.0, 7.0, 7.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let result = layer_norm_cpu(&values, &gamma, &beta, 1e-5);

        assert_eq!(result.normalized.len(), 3);
        for (i, v) in result.normalized.iter().enumerate() {
            assert!(
                v.abs() < 1e-3,
                "all-same values should normalize to ~0.0, got normalized[{i}]={v}"
            );
        }
    }

    #[test]
    fn layer_norm_cpu_nan_in_gamma_beta() {
        use super::layer_norm_cpu;

        let values = vec![1.0, 2.0];
        let gamma = vec![f64::NAN, 3.0];
        let beta = vec![1.0, f64::INFINITY];
        let result = layer_norm_cpu(&values, &gamma, &beta, 1e-5);

        assert_eq!(result.normalized.len(), 2);
        for (i, v) in result.normalized.iter().enumerate() {
            assert!(
                v.is_finite(),
                "normalized[{i}] should be finite even with NaN gamma/beta, got {v}"
            );
        }
    }

    #[test]
    fn layer_norm_frankentorch_stub_returns_error_without_feature() {
        #[cfg(not(feature = "gpu-frankentorch"))]
        {
            use super::layer_norm_frankentorch;

            let result = layer_norm_frankentorch(&[1.0, 2.0], &[1.0, 1.0], &[0.0, 0.0], 1e-5);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .contains("gpu-frankentorch feature not enabled")
            );
        }
    }

    #[test]
    fn compute_layer_norm_dispatch_uses_cpu_without_features() {
        use super::compute_layer_norm;

        let result = compute_layer_norm(&[1.0, 2.0, 3.0], &[1.0; 3], &[0.0; 3], 1e-5);

        #[cfg(not(feature = "gpu-frankentorch"))]
        assert!(!result.gpu_accelerated);

        assert_eq!(result.normalized.len(), 3);
        for v in &result.normalized {
            assert!(v.is_finite());
        }
    }

    // ===================================================================
    // bd-1r7.1: Frankentorch stub tests (feature disabled)
    // ===================================================================

    #[test]
    fn attention_scores_frankentorch_stub_returns_error_without_feature() {
        #[cfg(not(feature = "gpu-frankentorch"))]
        {
            use super::{AttentionKind, attention_scores_frankentorch};

            let result =
                attention_scores_frankentorch(&[1.0], &[1.0], AttentionKind::SelfAttention);
            assert!(result.is_err());
        }
    }

    #[test]
    fn embedding_lookup_frankentorch_stub_returns_error_without_feature() {
        #[cfg(not(feature = "gpu-frankentorch"))]
        {
            use super::{EmbeddingKind, embedding_lookup_frankentorch};

            let table = vec![vec![1.0]];
            let result = embedding_lookup_frankentorch(&[0], &table, EmbeddingKind::Token);
            assert!(result.is_err());
        }
    }

    #[test]
    fn vad_scores_frankentorch_stub_returns_error_without_feature() {
        #[cfg(not(feature = "gpu-frankentorch"))]
        {
            use super::vad_scores_frankentorch;

            let result = vad_scores_frankentorch(&[1.0, 2.0]);
            assert!(result.is_err());
        }
    }

    // ===================================================================
    // bd-1r7.2: InferenceGraphSpec tests
    // ===================================================================

    #[test]
    fn inference_graph_spec_new() {
        use super::{InferenceGraphPattern, InferenceGraphSpec};

        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        assert_eq!(spec.pattern, InferenceGraphPattern::NormalizeSoftmax);
        assert_eq!(spec.input_count, 1);
        assert_eq!(spec.input_dim, 64);
        assert!(spec.label.is_none());
    }

    #[test]
    fn inference_graph_spec_with_label() {
        use super::{InferenceGraphPattern, InferenceGraphSpec};

        let spec = InferenceGraphSpec::new(InferenceGraphPattern::AttentionBlock, 3, 128)
            .with_label("encoder_layer_0");
        assert_eq!(spec.label, Some("encoder_layer_0".to_owned()));
    }

    #[test]
    fn inference_graph_spec_equality() {
        use super::{InferenceGraphPattern, InferenceGraphSpec};

        let a = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 32);
        let b = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 32);
        assert_eq!(a, b);

        let c = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 64);
        assert_ne!(a, c);
    }

    #[test]
    fn inference_graph_spec_clone() {
        use super::{InferenceGraphPattern, InferenceGraphSpec};

        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 16)
            .with_label("test");
        let cloned = spec.clone();
        assert_eq!(spec, cloned);
    }

    // ===================================================================
    // bd-1r7.2: jit_cache_key tests
    // ===================================================================

    #[test]
    fn jit_cache_key_deterministic() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, jit_cache_key};

        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        let key1 = jit_cache_key(&spec);
        let key2 = jit_cache_key(&spec);
        assert_eq!(key1, key2, "cache key should be deterministic");
    }

    #[test]
    fn jit_cache_key_different_patterns_produce_different_keys() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, jit_cache_key};

        let spec_a = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        let spec_b = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 1, 64);
        assert_ne!(jit_cache_key(&spec_a), jit_cache_key(&spec_b));
    }

    #[test]
    fn jit_cache_key_different_dims_produce_different_keys() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, jit_cache_key};

        let spec_a = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        let spec_b = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 128);
        assert_ne!(jit_cache_key(&spec_a), jit_cache_key(&spec_b));
    }

    #[test]
    fn jit_cache_key_different_input_counts_produce_different_keys() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, jit_cache_key};

        let spec_a = InferenceGraphSpec::new(InferenceGraphPattern::AttentionBlock, 3, 64);
        let spec_b = InferenceGraphSpec::new(InferenceGraphPattern::AttentionBlock, 2, 64);
        assert_ne!(jit_cache_key(&spec_a), jit_cache_key(&spec_b));
    }

    #[test]
    fn jit_cache_key_contains_pattern_info() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, jit_cache_key};

        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        let key = jit_cache_key(&spec);
        assert!(
            key.contains("NormalizeSoftmax"),
            "key should contain pattern name: {key}"
        );
        assert!(key.contains("64"), "key should contain dimension: {key}");
    }

    // ===================================================================
    // bd-1r7.2: jit_compile_inference_graph tests
    // ===================================================================

    #[test]
    fn jit_compile_inference_graph_basic() {
        use super::{
            InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_compile_inference_graph,
        };

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);

        let result = jit_compile_inference_graph(&spec, &mut cache);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert_eq!(kernel.pattern, InferenceGraphPattern::NormalizeSoftmax);
        assert!(!kernel.kernel_id.is_empty());
    }

    #[test]
    fn jit_compile_inference_graph_caches_kernel() {
        use super::{
            InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_compile_inference_graph,
        };

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 32);

        let kernel1 = jit_compile_inference_graph(&spec, &mut cache).unwrap();
        let kernel2 = jit_compile_inference_graph(&spec, &mut cache).unwrap();

        // Second call should return the cached kernel with the same ID.
        assert_eq!(kernel1.kernel_id, kernel2.kernel_id);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn jit_compile_inference_graph_different_patterns() {
        use super::{
            InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_compile_inference_graph,
        };

        let mut cache = JitKernelCache::new();

        let spec_ns = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        let spec_la = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 64);
        let spec_ab = InferenceGraphSpec::new(InferenceGraphPattern::AttentionBlock, 3, 64);

        let k1 = jit_compile_inference_graph(&spec_ns, &mut cache).unwrap();
        let k2 = jit_compile_inference_graph(&spec_la, &mut cache).unwrap();
        let k3 = jit_compile_inference_graph(&spec_ab, &mut cache).unwrap();

        assert_eq!(cache.len(), 3);
        assert_ne!(k1.kernel_id, k2.kernel_id);
        assert_ne!(k2.kernel_id, k3.kernel_id);
    }

    // ===================================================================
    // bd-1r7.2: jit_execute_batch tests
    // ===================================================================

    #[test]
    fn jit_execute_batch_normalize_softmax() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 3);

        let batch = vec![vec![vec![1.0, 2.0, 3.0]], vec![vec![4.0, 5.0, 6.0]]];

        let results = jit_execute_batch(&spec, &batch, &mut cache);
        assert_eq!(results.len(), 2);

        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.pattern, InferenceGraphPattern::NormalizeSoftmax);
            assert_eq!(r.values.len(), 3);
            let sum: f64 = r.values.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "batch[{i}] should sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn jit_execute_batch_linear_activation() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 2);

        let batch = vec![
            vec![vec![1.0, -1.0], vec![2.0, 2.0]],
            vec![vec![3.0, 0.5], vec![1.0, 1.0]],
        ];

        let results = jit_execute_batch(&spec, &batch, &mut cache);
        assert_eq!(results.len(), 2);

        for r in &results {
            assert_eq!(r.pattern, InferenceGraphPattern::LinearActivation);
            for v in &r.values {
                assert!(*v >= 0.0, "ReLU output should be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn jit_execute_batch_attention_block() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::AttentionBlock, 3, 3);

        let batch = vec![vec![
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![5.0, 10.0, 15.0],
        ]];

        let results = jit_execute_batch(&spec, &batch, &mut cache);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pattern, InferenceGraphPattern::AttentionBlock);
        assert_eq!(results[0].values.len(), 3);

        for v in &results[0].values {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[test]
    fn jit_execute_batch_empty_batch() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 4);

        let results = jit_execute_batch(&spec, &[], &mut cache);
        assert!(results.is_empty());
    }

    #[test]
    fn jit_execute_batch_populates_cache() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();
        assert!(cache.is_empty());

        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 2);
        let batch = vec![vec![vec![1.0, 2.0]]];

        jit_execute_batch(&spec, &batch, &mut cache);
        assert!(
            !cache.is_empty(),
            "cache should be populated after batch execution"
        );
    }

    #[test]
    fn jit_execute_batch_uses_compiled_cache() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();
        let spec = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 3);

        let batch = vec![vec![vec![1.0, 2.0, 3.0]]];

        // jit_execute_batch compiles the graph first, so the internal
        // jit_inference call already sees the cached kernel.
        let results1 = jit_execute_batch(&spec, &batch, &mut cache);
        assert!(
            results1[0].cache_hit,
            "batch execution compiles first, so inference sees a cached kernel"
        );

        // Second call: still a cache hit.
        let results2 = jit_execute_batch(&spec, &batch, &mut cache);
        assert!(results2[0].cache_hit);

        // Values should be identical between runs.
        assert_eq!(results1[0].values, results2[0].values);
    }

    #[test]
    fn jit_execute_batch_multiple_patterns_in_sequence() {
        use super::{InferenceGraphPattern, InferenceGraphSpec, JitKernelCache, jit_execute_batch};

        let mut cache = JitKernelCache::new();

        let spec_ns = InferenceGraphSpec::new(InferenceGraphPattern::NormalizeSoftmax, 1, 2);
        let spec_la = InferenceGraphSpec::new(InferenceGraphPattern::LinearActivation, 2, 2);

        let batch_ns = vec![vec![vec![1.0, 2.0]]];
        let batch_la = vec![vec![vec![1.0, 2.0], vec![0.5, 0.5]]];

        let r_ns = jit_execute_batch(&spec_ns, &batch_ns, &mut cache);
        let r_la = jit_execute_batch(&spec_la, &batch_la, &mut cache);

        assert_eq!(r_ns.len(), 1);
        assert_eq!(r_la.len(), 1);
        assert_eq!(r_ns[0].pattern, InferenceGraphPattern::NormalizeSoftmax);
        assert_eq!(r_la[0].pattern, InferenceGraphPattern::LinearActivation);
    }

    // ── bd-1r7.3: benchmark harness tests ──

    #[test]
    fn benchmark_softmax_produces_valid_result() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_softmax(128);

        assert!(result.operation.contains("softmax"));
        assert!(result.operation.contains("128"));
        // GPU not available, so gpu_time_us should be None.
        assert!(result.gpu_time_us.is_none());
        assert_eq!(result.speedup_ratio, 0.0);
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_layer_norm_produces_valid_result() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_layer_norm(256);

        assert!(result.operation.contains("layer_norm"));
        assert!(result.operation.contains("256"));
        assert!(result.gpu_time_us.is_none());
        assert_eq!(result.speedup_ratio, 0.0);
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_attention_produces_valid_result() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_attention(64, 32);

        assert!(result.operation.contains("attention"));
        assert!(result.operation.contains("64"));
        assert!(result.operation.contains("32"));
        assert!(result.gpu_time_us.is_none());
        assert_eq!(result.speedup_ratio, 0.0);
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_run_all_returns_expected_count() {
        let mut harness = super::BenchmarkHarness::new();
        let results = harness.run_all();

        // 3 softmax + 3 layer_norm + 2 attention = 8.
        assert_eq!(results.len(), 8);

        // Verify each result has a non-empty operation name.
        for r in &results {
            assert!(!r.operation.is_empty());
        }
    }

    #[test]
    fn benchmark_recommend_backend_without_gpu_returns_cpu() {
        let mut harness = super::BenchmarkHarness::new();
        let _ = harness.run_all();

        let recommendation = harness.recommend_backend();
        assert_eq!(recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_recommend_backend_empty_returns_cpu() {
        let harness = super::BenchmarkHarness::new();
        assert_eq!(
            harness.recommend_backend(),
            super::AccelRecommendation::UseCpu
        );
    }

    #[test]
    fn benchmark_report_serialization_roundtrip() {
        let mut harness = super::BenchmarkHarness::new();
        harness.benchmark_softmax(32);
        harness.benchmark_layer_norm(32);

        let report = harness.report();

        // Serialize to JSON.
        let json_str = serde_json::to_string_pretty(&report).expect("serialize report");

        // Deserialize back.
        let roundtripped: super::BenchmarkReport =
            serde_json::from_str(&json_str).expect("deserialize report");

        assert_eq!(roundtripped.results.len(), report.results.len());
        assert_eq!(roundtripped.overall, report.overall);
        assert_eq!(roundtripped.notes, report.notes);

        // Verify the JSON contains expected fields.
        let value: serde_json::Value =
            serde_json::from_str(&json_str).expect("parse as json value");
        assert!(value.get("results").is_some());
        assert!(value.get("overall").is_some());
        assert!(value.get("notes").is_some());
    }

    #[test]
    fn benchmark_report_notes_mention_gpu_unavailable() {
        let mut harness = super::BenchmarkHarness::new();
        harness.benchmark_softmax(16);

        let report = harness.report();
        assert!(!report.notes.is_empty());
        let combined = report.notes.join(" ");
        assert!(
            combined.contains("GPU") || combined.contains("gpu"),
            "notes should mention GPU unavailability: {:?}",
            report.notes
        );
    }

    #[test]
    fn benchmark_result_from_timings_gpu_faster() {
        let result = super::benchmark_result_from_timings(
            "test_op".to_owned(),
            1000,      // CPU: 1000us
            Some(500), // GPU: 500us -> ratio 2.0, GPU wins
        );
        assert_eq!(result.recommendation, super::AccelRecommendation::UseGpu);
        assert!((result.speedup_ratio - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_result_from_timings_cpu_faster() {
        let result = super::benchmark_result_from_timings(
            "test_op".to_owned(),
            500,        // CPU: 500us
            Some(1000), // GPU: 1000us -> ratio 0.5, CPU wins
        );
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
        assert!((result.speedup_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_result_from_timings_either_fine() {
        let result = super::benchmark_result_from_timings(
            "test_op".to_owned(),
            1000,       // CPU: 1000us
            Some(1000), // GPU: 1000us -> ratio 1.0, within 10%
        );
        assert_eq!(
            result.recommendation,
            super::AccelRecommendation::EitherFine
        );
        assert!((result.speedup_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_result_from_timings_no_gpu() {
        let result = super::benchmark_result_from_timings("test_op".to_owned(), 1000, None);
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
        assert_eq!(result.speedup_ratio, 0.0);
        assert!(result.gpu_time_us.is_none());
    }

    #[test]
    fn benchmark_result_from_timings_zero_gpu_time() {
        let result = super::benchmark_result_from_timings(
            "test_op".to_owned(),
            1000,
            Some(0), // zero GPU time treated as unavailable
        );
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
        assert_eq!(result.speedup_ratio, 0.0);
    }

    #[test]
    fn accel_recommendation_display() {
        assert_eq!(super::AccelRecommendation::UseCpu.to_string(), "use_cpu");
        assert_eq!(super::AccelRecommendation::UseGpu.to_string(), "use_gpu");
        assert_eq!(
            super::AccelRecommendation::EitherFine.to_string(),
            "either_fine"
        );
    }

    #[test]
    fn benchmark_harness_default_is_empty() {
        let harness = super::BenchmarkHarness::default();
        assert_eq!(
            harness.recommend_backend(),
            super::AccelRecommendation::UseCpu
        );
    }

    #[test]
    fn benchmark_recommend_backend_majority_gpu() {
        // Manually construct results where GPU wins majority.
        let harness = super::BenchmarkHarness {
            results: vec![
                super::BenchmarkResult {
                    operation: "a".to_owned(),
                    cpu_time_us: 1000,
                    gpu_time_us: Some(100),
                    speedup_ratio: 10.0,
                    recommendation: super::AccelRecommendation::UseGpu,
                },
                super::BenchmarkResult {
                    operation: "b".to_owned(),
                    cpu_time_us: 1000,
                    gpu_time_us: Some(200),
                    speedup_ratio: 5.0,
                    recommendation: super::AccelRecommendation::UseGpu,
                },
                super::BenchmarkResult {
                    operation: "c".to_owned(),
                    cpu_time_us: 100,
                    gpu_time_us: Some(500),
                    speedup_ratio: 0.2,
                    recommendation: super::AccelRecommendation::UseCpu,
                },
            ],
        };
        assert_eq!(
            harness.recommend_backend(),
            super::AccelRecommendation::UseGpu
        );
    }

    #[test]
    fn benchmark_recommend_backend_evenly_split_returns_either() {
        let harness = super::BenchmarkHarness {
            results: vec![
                super::BenchmarkResult {
                    operation: "a".to_owned(),
                    cpu_time_us: 1000,
                    gpu_time_us: Some(100),
                    speedup_ratio: 10.0,
                    recommendation: super::AccelRecommendation::UseGpu,
                },
                super::BenchmarkResult {
                    operation: "b".to_owned(),
                    cpu_time_us: 100,
                    gpu_time_us: Some(1000),
                    speedup_ratio: 0.1,
                    recommendation: super::AccelRecommendation::UseCpu,
                },
            ],
        };
        assert_eq!(
            harness.recommend_backend(),
            super::AccelRecommendation::EitherFine,
        );
    }

    #[test]
    fn benchmark_run_all_clears_previous_results() {
        let mut harness = super::BenchmarkHarness::new();
        harness.benchmark_softmax(16);
        assert_eq!(harness.results.len(), 1);

        let results = harness.run_all();
        // run_all clears previous and runs fresh set.
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn benchmark_softmax_zero_size() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_softmax(0);
        assert!(result.operation.contains("0"));
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_layer_norm_zero_size() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_layer_norm(0);
        assert!(result.operation.contains("0"));
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_attention_zero_dimensions() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_attention(0, 0);
        assert_eq!(result.recommendation, super::AccelRecommendation::UseCpu);
    }

    #[test]
    fn benchmark_report_on_empty_harness() {
        let harness = super::BenchmarkHarness::new();
        let report = harness.report();
        assert_eq!(report.overall, super::AccelRecommendation::UseCpu);
        assert!(
            !report.notes.is_empty(),
            "empty harness should emit GPU-unavailable note"
        );
        assert!(report.results.is_empty());
    }

    #[test]
    fn benchmark_result_from_timings_exact_boundary_values() {
        // ratio == 1.1 exactly: not > 1.1, so EitherFine
        let result =
            super::benchmark_result_from_timings("boundary_high".to_owned(), 1100, Some(1000));
        assert_eq!(
            result.recommendation,
            super::AccelRecommendation::EitherFine
        );

        // ratio == 0.9 exactly: not < 0.9, so EitherFine
        let result2 =
            super::benchmark_result_from_timings("boundary_low".to_owned(), 900, Some(1000));
        assert_eq!(
            result2.recommendation,
            super::AccelRecommendation::EitherFine
        );
    }

    #[test]
    fn benchmark_recommend_backend_all_either_fine_returns_either() {
        let harness = super::BenchmarkHarness {
            results: vec![
                super::BenchmarkResult {
                    operation: "a".to_owned(),
                    cpu_time_us: 1000,
                    gpu_time_us: Some(1000),
                    speedup_ratio: 1.0,
                    recommendation: super::AccelRecommendation::EitherFine,
                },
                super::BenchmarkResult {
                    operation: "b".to_owned(),
                    cpu_time_us: 950,
                    gpu_time_us: Some(1000),
                    speedup_ratio: 0.95,
                    recommendation: super::AccelRecommendation::EitherFine,
                },
            ],
        };
        assert_eq!(
            harness.recommend_backend(),
            super::AccelRecommendation::EitherFine
        );
    }

    #[test]
    fn jit_attention_block_cpu_value_is_shortest() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![10.0, 20.0]; // shortest
        let result = super::jit_attention_block_cpu(&q, &k, &v);
        assert_eq!(result.len(), 2);
        for val in &result {
            assert!(val.is_finite() && *val >= 0.0);
        }
    }

    #[test]
    fn jit_cache_key_ignores_label_field() {
        let spec_no_label =
            super::InferenceGraphSpec::new(super::InferenceGraphPattern::NormalizeSoftmax, 1, 64);
        let spec_with_label =
            super::InferenceGraphSpec::new(super::InferenceGraphPattern::NormalizeSoftmax, 1, 64)
                .with_label("encoder_attn");
        let spec_different_label =
            super::InferenceGraphSpec::new(super::InferenceGraphPattern::NormalizeSoftmax, 1, 64)
                .with_label("decoder_attn");

        assert_eq!(
            super::jit_cache_key(&spec_no_label),
            super::jit_cache_key(&spec_with_label)
        );
        assert_eq!(
            super::jit_cache_key(&spec_with_label),
            super::jit_cache_key(&spec_different_label)
        );
    }

    // -- bd-247: accelerate.rs edge-case tests pass 2 --

    #[test]
    fn apply_with_token_cancelled_returns_none_backend() {
        use crate::orchestrator::CancellationToken;
        use std::time::Duration;

        let mut result = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "hello".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "hello".to_owned(),
                confidence: Some(0.8),
                speaker: None,
            }],
            acceleration: None,
            raw_output: json!({}),
            artifact_paths: vec![],
        };
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));
        let report = super::apply_with_token(&mut result, Some(&token));
        assert_eq!(report.backend, AccelerationBackend::None);
        assert_eq!(report.input_values, 0);
        assert!(!report.normalized_confidences);
        assert!(report.notes[0].contains("cancelled"));
        // Side effect: result.acceleration should be set.
        assert!(result.acceleration.is_some());
    }

    #[test]
    fn normalize_cpu_single_value_at_epsilon_boundary() {
        // f64::EPSILON ≈ 2.22e-16, which is > 0.0 and finite, so it passes the filter.
        // But safe_sum = f64::EPSILON ≈ 2.22e-16 <= f64::EPSILON, so uniform branch fires.
        let result = super::normalize_cpu(&[f64::EPSILON]);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 1.0).abs() < 1e-9,
            "single epsilon value should produce uniform [1.0], got {:?}",
            result
        );
    }

    #[test]
    fn layer_norm_cpu_zero_epsilon_constant_input_stays_finite() {
        let values = vec![5.0, 5.0, 5.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let result = super::layer_norm_cpu(&values, &gamma, &beta, 0.0);
        assert!(!result.gpu_accelerated);
        for v in &result.normalized {
            assert!(
                v.is_finite() && v.abs() < 1e-9,
                "expected finite near-zero output from zero-epsilon constant input, got {v}"
            );
        }
    }

    #[test]
    fn layer_norm_cpu_negative_epsilon_is_clamped() {
        let values = vec![2.0, 2.0, 2.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let result = super::layer_norm_cpu(&values, &gamma, &beta, -1.0);

        assert!(!result.gpu_accelerated);
        for v in &result.normalized {
            assert!(
                v.is_finite() && v.abs() < 1e-9,
                "negative epsilon should be clamped to a stable floor, got {v}"
            );
        }
    }

    #[test]
    fn benchmark_attention_d_model_exceeds_seq_len() {
        let mut harness = super::BenchmarkHarness::new();
        let result = harness.benchmark_attention(16, 128);
        assert!(
            result.operation.contains("seq_len=16"),
            "operation should mention seq_len=16, got: {}",
            result.operation
        );
        assert!(
            result.operation.contains("d_model=128"),
            "operation should mention d_model=128, got: {}",
            result.operation
        );
    }

    #[test]
    fn benchmark_report_with_gpu_result_omits_gpu_unavailable_note() {
        let mut harness = super::BenchmarkHarness::new();
        // Manually insert a result that simulates GPU availability.
        harness.results.push(super::BenchmarkResult {
            operation: "softmax(n=64)".to_owned(),
            cpu_time_us: 100,
            gpu_time_us: Some(50),
            speedup_ratio: 2.0,
            recommendation: super::AccelRecommendation::UseGpu,
        });
        let report = harness.report();
        for note in &report.notes {
            assert!(
                !note.contains("GPU backend not available"),
                "report with GPU results should not include GPU-unavailable note"
            );
        }
    }
}
