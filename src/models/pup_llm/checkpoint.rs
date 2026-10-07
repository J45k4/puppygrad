//! Safetensors/config bindings for .pup LLM providers (GPT-2 and Qwen3).
use crate::compiler::{
    cpu::Tensor,
    pop::{DType, Scalar},
    source::{Context, TensorSpec},
};
use std::{
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub struct Checkpoint {
    pub context: Context,
    pub inputs: Vec<Tensor>,
    pub model_type: Option<String>,
}
impl Checkpoint {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut context = config_context(dir)?;
        let model_type = model_type(dir)?;
        let tied = tied_embeddings(&context);
        let (mut file, metadata, payload_start) = checkpoint_file(dir)?;
        let tensors = metadata.tensors();
        let mut names: Vec<_> = tensors.keys().collect();
        names.sort();
        let mut inputs = vec![Tensor::I32(Vec::new().into())];
        for name in names {
            let Some(key) = weight_key(name) else {
                continue;
            };
            if tied && key == "lm_head.weight" {
                continue;
            }
            let info = tensors[name];
            let (start, end) = info.data_offsets;
            file.seek(SeekFrom::Start(payload_start + start as u64))?;
            // Do not keep a second full checkpoint in RAM during conversion.
            let mut bytes = vec![0; end - start];
            file.read_exact(&mut bytes)?;
            let tensor =
                safetensors::tensor::TensorView::new(info.dtype, info.shape.clone(), &bytes)?;
            let values = crate::models::safetensors::tensor_data_as_f32(name, &tensor)?;
            bind_weight(&mut context, key, inputs.len(), tensor.shape())?;
            inputs.push(Tensor::F32(values.into()));
        }
        bind_tied_head(&mut context)?;
        Ok(Self {
            context,
            inputs,
            model_type,
        })
    }
    /// Resolve shapes and config without reading or allocating weight values.
    pub fn metadata_context(dir: &Path, sequence_length: usize) -> Result<Context> {
        if sequence_length == 0 {
            return Err("sequence length must be greater than zero".into());
        }
        let mut context = config_context(dir)?;
        let tied = tied_embeddings(&context);
        let (_, metadata, _) = checkpoint_file(dir)?;
        let tensors = metadata.tensors();
        let mut names: Vec<_> = tensors.keys().collect();
        names.sort();
        let mut slot = 1;
        for name in names {
            let Some(key) = weight_key(name) else {
                continue;
            };
            if tied && key == "lm_head.weight" {
                continue;
            }
            let tensor = tensors[name];
            if !matches!(
                tensor.dtype,
                safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
            ) {
                return Err(format!(
                    "unsupported checkpoint weight dtype for {name}: {:?}",
                    tensor.dtype
                )
                .into());
            }
            bind_weight(&mut context, key, slot, &tensor.shape)?;
            slot += 1;
        }
        bind_tied_head(&mut context)?;
        context.tensors.insert(
            "tokens".into(),
            TensorSpec {
                slot: 0,
                dtype: DType::I32,
                shape: vec![sequence_length],
            },
        );
        Ok(context)
    }
    pub fn bind_tokens(&mut self, tokens: &[usize]) -> Result<()> {
        if tokens.is_empty() {
            return Err("prompt must contain at least one token".into());
        }
        let ids = tokens
            .iter()
            .map(|&n| i32::try_from(n))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        self.inputs[0] = Tensor::I32(ids.into());
        self.context.tensors.insert(
            "tokens".into(),
            TensorSpec {
                slot: 0,
                dtype: DType::I32,
                shape: vec![tokens.len()],
            },
        );
        Ok(())
    }
}

fn checkpoint_file(dir: &Path) -> Result<(std::fs::File, safetensors::tensor::Metadata, u64)> {
    let mut file = std::fs::File::open(dir.join("model.safetensors"))?;
    let mut size = [0; 8];
    file.read_exact(&mut size)?;
    let size = u64::from_le_bytes(size);
    let file_len = file.metadata()?.len();
    if size > 100_000_000 || size > file_len.saturating_sub(8) {
        return Err("invalid safetensors header length".into());
    }
    let mut header = vec![0; usize::try_from(size)?];
    file.read_exact(&mut header)?;
    let metadata: safetensors::tensor::Metadata = serde_json::from_slice(&header)?;
    if size
        .checked_add(8)
        .and_then(|n| n.checked_add(metadata.data_len() as u64))
        != Some(file_len)
    {
        return Err("safetensors payload size does not match metadata".into());
    }
    Ok((file, metadata, size + 8))
}

fn config_context(dir: &Path) -> Result<Context> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let mut context = Context::default();
    for (key, value) in config.as_object().ok_or("config must be a JSON object")? {
        let scalar = if let Some(b) = value.as_bool() {
            Some(Scalar::Bool(b))
        } else if let Some(i) = value.as_i64() {
            Some(Scalar::Int(i))
        } else {
            value.as_f64().map(Scalar::float)
        };
        if let Some(scalar) = scalar {
            context.constants.insert(key.clone(), scalar);
        }
    }
    // Normalize only the runtime's context-limit field; model math keeps its
    // original named config values.
    if !context.constants.contains_key("n_positions") {
        if let Some(&limit) = context.constants.get("max_position_embeddings") {
            context.constants.insert("n_positions".into(), limit);
        }
    }
    // A generic retained-buffer capacity. Providers specialize it to the
    // requested stream; metadata-only emission uses a modest default.
    let limit = match context.constants.get("n_positions") {
        Some(Scalar::Int(n)) if *n > 0 => *n,
        _ => 512,
    };
    context
        .constants
        .insert("buffer_capacity".into(), Scalar::Int(limit.min(512)));
    Ok(context)
}
fn model_type(dir: &Path) -> Result<Option<String>> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    Ok(config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned))
}
fn tied_embeddings(context: &Context) -> bool {
    context.constants.get("tie_word_embeddings") == Some(&Scalar::Bool(true))
        && context.constants.contains_key("num_hidden_layers")
}
fn bind_tied_head(context: &mut Context) -> Result<()> {
    if tied_embeddings(context) {
        let embedding = context
            .tensors
            .get("model.embed_tokens.weight")
            .ok_or("tied embeddings require model.embed_tokens.weight")?
            .clone();
        context.tensors.insert("lm_head.weight".into(), embedding);
    }
    Ok(())
}

fn weight_key(name: &str) -> Option<&str> {
    if name.ends_with(".attn.bias") || name.ends_with(".attn.masked_bias") {
        return None;
    }
    Some(name.strip_prefix("transformer.").unwrap_or(name))
}

fn bind_weight(context: &mut Context, key: &str, slot: usize, shape: &[usize]) -> Result<()> {
    if context.tensors.contains_key(key) {
        return Err(format!("duplicate normalized weight {key}").into());
    }
    context.tensors.insert(
        key.into(),
        TensorSpec {
            slot,
            dtype: DType::F32,
            shape: shape.to_vec(),
        },
    );
    Ok(())
}
