//! Phase 0 artifact byte census (bd-3be3.5): exact per-tensor-class byte
//! composition of whisper ggml files and the sortformer safetensors, so
//! phases 3-4 of bd-3be3 order their quantization effort by measured mass
//! instead of assumption (frankentts's census found 47% of their artifact in
//! ONE cold tensor — this answers whether turbo has such a lever).
//!
//! Usage: `cargo run --release --example artifact_census -- <path>...`
//! Dispatch is by extension: `.bin` parses as whisper ggml, `.safetensors`
//! as sortformer safetensors. For each file: an aligned per-class table
//! (bytes descending), the ten largest tensors, and one NDJSON line per
//! class for machine consumption.

use std::collections::BTreeMap;

use franken_whisper::native_engine::GgmlDType;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::weights::SafetensorsFile;

/// On-disk payload bytes for the dtypes our artifacts actually contain.
/// Mirrors the parser's own block math (`ggml_byte_len`); anything else is a
/// hard error rather than a silent estimate.
fn ggml_payload_bytes(dtype: GgmlDType, n: usize) -> Result<usize, String> {
    match dtype {
        GgmlDType::F32 => Ok(n * 4),
        GgmlDType::F16 => Ok(n * 2),
        GgmlDType::Q8_0 => {
            if n % 32 != 0 {
                return Err(format!("q8_0 element count {n} not a multiple of 32"));
            }
            Ok(n / 32 * 34)
        }
        other => Err(format!("unhandled dtype {other:?} — extend the census")),
    }
}

/// Whisper ggml tensor taxonomy. Norm tensors carry `_ln.` / `ln_post.` /
/// `decoder.ln.` names, so they are matched before the attention/MLP
/// substrings they would otherwise shadow.
fn whisper_class(name: &str) -> &'static str {
    let enc = name.starts_with("encoder.");
    if name.contains("_ln.") || name.contains(".ln_post.") || name.starts_with("decoder.ln.") {
        return if enc { "enc.ln" } else { "dec.ln" };
    }
    if name.starts_with("encoder.conv") {
        return "enc.conv_stem";
    }
    if name == "encoder.positional_embedding" {
        return "enc.pos";
    }
    if name == "decoder.positional_embedding" {
        return "dec.pos";
    }
    if name.starts_with("decoder.token_embedding") {
        return "dec.tok_embed";
    }
    if name.contains(".cross_attn.") {
        return "dec.cross_attn";
    }
    if name.contains(".attn.") {
        return if enc { "enc.attn" } else { "dec.self_attn" };
    }
    if name.contains(".mlp.") {
        return if enc { "enc.mlp" } else { "dec.mlp" };
    }
    "other"
}

/// Sortformer safetensors taxonomy: fastconformer (`fc.*`), the four-speaker
/// transformer head stack (`tf.*`), and the sortformer projection heads.
fn sortformer_class(name: &str) -> &'static str {
    if name.starts_with("preprocessor.") {
        return "preproc";
    }
    if name.starts_with("encoder.pre_encode") {
        return "fc.pre_encode";
    }
    if name.starts_with("encoder.pos_enc") {
        return "fc.pos_enc";
    }
    if name.starts_with("encoder.layers.") {
        if name.contains(".self_attn.") {
            return "fc.self_attn";
        }
        if name.contains(".conv.") {
            return "fc.conv";
        }
        if name.contains(".feed_forward") {
            return "fc.feed_forward";
        }
        if name.contains(".norm") {
            return "fc.norm";
        }
        return "fc.other";
    }
    if name.starts_with("transformer_encoder.") {
        if name.contains(".first_sub_layer.") {
            return "tf.attn";
        }
        if name.contains(".second_sub_layer.") {
            return "tf.mlp";
        }
        if name.contains(".layer_norm") {
            return "tf.norm";
        }
        return "tf.other";
    }
    if name.starts_with("sortformer_modules.") {
        return "heads";
    }
    "other"
}

#[derive(Default)]
struct ClassStat {
    tensors: usize,
    params: usize,
    bytes: usize,
}

struct Row {
    name: String,
    dtype: String,
    params: usize,
    bytes: usize,
}

fn report(label: &str, path: &str, file_bytes: u64, rows: &[Row], class_of: fn(&str) -> &'static str) {
    let mut classes: BTreeMap<&'static str, ClassStat> = BTreeMap::new();
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut payload = 0usize;
    for r in rows {
        let s = classes.entry(class_of(&r.name)).or_default();
        s.tensors += 1;
        s.params += r.params;
        s.bytes += r.bytes;
        *dtypes.entry(r.dtype.clone()).or_default() += r.bytes;
        payload += r.bytes;
    }

    println!("\n== {label}: {path}");
    println!(
        "   file {file_bytes} B ({:.1} MB) | tensor payload {payload} B ({:.1} MB) | header/vocab/filters {} B | {} tensors",
        file_bytes as f64 / 1e6,
        payload as f64 / 1e6,
        file_bytes as i64 - payload as i64,
        rows.len(),
    );
    let dtype_line: Vec<String> = dtypes
        .iter()
        .map(|(d, b)| format!("{d} {:.1} MB", *b as f64 / 1e6))
        .collect();
    println!("   by dtype: {}", dtype_line.join(" | "));

    let mut ordered: Vec<(&&str, &ClassStat)> = classes.iter().collect();
    ordered.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
    println!("   {:<14} {:>7} {:>13} {:>13} {:>8} {:>6}", "class", "tensors", "params", "bytes", "MB", "%");
    for (class, s) in &ordered {
        println!(
            "   {:<14} {:>7} {:>13} {:>13} {:>8.1} {:>5.1}%",
            class,
            s.tensors,
            s.params,
            s.bytes,
            s.bytes as f64 / 1e6,
            s.bytes as f64 * 100.0 / payload as f64,
        );
    }

    let mut top: Vec<&Row> = rows.iter().collect();
    top.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    println!("   largest tensors:");
    for r in top.iter().take(10) {
        println!(
            "     {:>10} B ({:>4.1}%) {:<6} {}",
            r.bytes,
            r.bytes as f64 * 100.0 / payload as f64,
            r.dtype,
            r.name
        );
    }

    for (class, s) in &ordered {
        println!(
            "{{\"event\":\"artifact_census.class\",\"file\":\"{label}\",\"class\":\"{class}\",\"tensors\":{},\"params\":{},\"bytes\":{},\"pct\":{:.2}}}",
            s.tensors,
            s.params,
            s.bytes,
            s.bytes as f64 * 100.0 / payload as f64,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: artifact_census <path.bin|path.safetensors>...".into());
    }
    for path in &args {
        let file_bytes = std::fs::metadata(path)?.len();
        if path.ends_with(".safetensors") {
            let file = SafetensorsFile::load(std::path::Path::new(path))?;
            let names: Vec<String> = file.names().map(str::to_owned).collect();
            let mut rows = Vec::with_capacity(names.len());
            for name in names {
                let shape = file.shape(&name)?;
                let params: usize = shape.iter().product();
                let dtype = file.dtype_name(&name)?;
                let per_elem = match dtype {
                    "F32" => 4,
                    "F16" | "BF16" => 2,
                    other => return Err(format!("{name}: unhandled dtype {other}").into()),
                };
                rows.push(Row {
                    name,
                    dtype: dtype.to_owned(),
                    params,
                    bytes: params * per_elem,
                });
            }
            report("safetensors", path, file_bytes, &rows, sortformer_class);
        } else {
            let model = GgmlModel::load(std::path::Path::new(path))?;
            let names: Vec<String> = model.tensor_names().map(str::to_owned).collect();
            let mut rows = Vec::with_capacity(names.len());
            for name in names {
                let entry = model
                    .tensor(&name)
                    .ok_or_else(|| format!("tensor {name} vanished from directory"))?;
                let params = entry.n_elements();
                let bytes = ggml_payload_bytes(entry.dtype, params).map_err(|e| format!("{name}: {e}"))?;
                rows.push(Row {
                    name,
                    dtype: format!("{:?}", entry.dtype),
                    params,
                    bytes,
                });
            }
            report("ggml", path, file_bytes, &rows, whisper_class);
        }
    }
    Ok(())
}
