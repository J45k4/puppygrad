//! Safetensors/config bindings for .pup LLM providers (GPT-2 and Qwen3).
use crate::compiler::{
    cpu::Tensor,
    pop::{DType, Scalar},
    source::{Context, TensorSpec},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek, SeekFrom},
    path::{Component, Path},
    sync::Arc,
    time::{Duration, Instant},
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
        let mut files = CheckpointFiles::open(dir)?;
        let mut inputs = vec![Tensor::I32(Vec::new().into())];
        let mut scratch = Vec::new();
        let mut timing = LoadTiming::default();
        for (name, (shard, info)) in &files.tensors {
            let Some(key) = weight_key(name) else {
                continue;
            };
            if tied && key == "lm_head.weight" {
                continue;
            }
            let (start, end) = info.data_offsets;
            let shard = &mut files.shards[*shard];
            shard
                .file
                .seek(SeekFrom::Start(shard.payload_start + start as u64))?;
            let values = read_weight(
                name,
                info.dtype,
                end - start,
                &mut shard.file,
                &mut scratch,
                &mut timing,
            )?;
            bind_weight(&mut context, key, inputs.len(), &info.shape)?;
            inputs.push(Tensor::F32(values));
        }
        bind_tied_head(&mut context)?;
        let values: usize = inputs.iter().skip(1).map(Tensor::len).sum();
        crate::progress::emit(format!(
            "Checkpoint weights: read {:.3}s · convert {:.3}s · {:.2} GiB F32",
            timing.read.as_secs_f64(),
            timing.convert.as_secs_f64(),
            values as f64 * 4. / 1_073_741_824.,
        ));
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
        let files = CheckpointFiles::open(dir)?;
        let mut slot = 1;
        for (name, (_, tensor)) in &files.tensors {
            let Some(key) = weight_key(name) else {
                continue;
            };
            if tied && key == "lm_head.weight" {
                continue;
            }
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

// Read bounded chunks directly into the final shared allocation. Converting to
// Vec<f32> first and then Arc<[f32]> copies every weight and touches extra pages.
const READ_CHUNK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Default)]
struct LoadTiming {
    read: Duration,
    convert: Duration,
}

fn read_weight(
    name: &str,
    dtype: safetensors::Dtype,
    byte_len: usize,
    reader: &mut impl Read,
    scratch: &mut Vec<u8>,
    timing: &mut LoadTiming,
) -> Result<Arc<[f32]>> {
    use safetensors::Dtype;
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => {
            return Err(format!("unsupported checkpoint weight dtype for {name}: {dtype:?}").into())
        }
    };
    if !byte_len.is_multiple_of(width) {
        return Err(format!("invalid checkpoint weight byte length for {name}: {byte_len}").into());
    }
    let started = Instant::now();
    let mut values = Arc::<[f32]>::new_uninit_slice(byte_len / width);
    let buffer_len = byte_len.min(READ_CHUNK_BYTES);
    if scratch.len() < buffer_len {
        scratch.resize(buffer_len, 0);
    }
    timing.convert += started.elapsed();
    for output in Arc::get_mut(&mut values)
        .unwrap()
        .chunks_mut(READ_CHUNK_BYTES / width)
    {
        let data = &mut scratch[..output.len() * width];
        let started = Instant::now();
        reader.read_exact(data)?;
        timing.read += started.elapsed();
        let started = Instant::now();
        match dtype {
            Dtype::F32 => {
                for (slot, bytes) in output.iter_mut().zip(data.as_chunks::<4>().0.iter()) {
                    slot.write(f32::from_le_bytes(*bytes));
                }
            }
            Dtype::F16 => {
                for (slot, bytes) in output.iter_mut().zip(data.as_chunks::<2>().0.iter()) {
                    slot.write(half::f16::from_bits(u16::from_le_bytes(*bytes)).to_f32());
                }
            }
            Dtype::BF16 => {
                for (slot, bytes) in output.iter_mut().zip(data.as_chunks::<2>().0.iter()) {
                    let bits = u16::from_le_bytes(*bytes);
                    // BF16 is the high 16 bits of F32. Preserve the half crate's
                    // quieting of signaling NaNs, including sign and payload.
                    let bits = if bits & 0x7fff > 0x7f80 {
                        bits | 0x0040
                    } else {
                        bits
                    };
                    slot.write(f32::from_bits(u32::from(bits) << 16));
                }
            }
            _ => unreachable!(),
        }
        timing.convert += started.elapsed();
    }
    // SAFETY: every element was initialized above. Read failures return before
    // this point and safely drop the allocation of MaybeUninit<f32> instead.
    Ok(unsafe { values.assume_init() })
}

struct Shard {
    file: std::fs::File,
    payload_start: u64,
}
struct CheckpointFiles {
    shards: Vec<Shard>,
    tensors: BTreeMap<String, (usize, safetensors::tensor::TensorInfo)>,
}
impl CheckpointFiles {
    fn open(dir: &Path) -> Result<Self> {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let index = if dir.join("model.safetensors").is_file() {
            None
        } else {
            let index: Index =
                serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
            if index.weight_map.is_empty() {
                return Err("checkpoint shard index has no weights".into());
            }
            for name in index.weight_map.values() {
                if name.is_empty()
                    || !name.ends_with(".safetensors")
                    || !Path::new(name)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_)))
                {
                    return Err(format!("invalid checkpoint shard path: {name}").into());
                }
            }
            Some(index)
        };
        let paths: BTreeSet<_> = index.as_ref().map_or_else(
            || BTreeSet::from(["model.safetensors".to_owned()]),
            |index| index.weight_map.values().cloned().collect(),
        );
        let mut files = Self {
            shards: Vec::new(),
            tensors: BTreeMap::new(),
        };
        for path in paths {
            let (file, metadata, payload_start) = checkpoint_file(&dir.join(&path))?;
            let shard = files.shards.len();
            for (name, info) in metadata.tensors() {
                if let Some(index) = &index {
                    if index.weight_map.get(&name) != Some(&path) {
                        return Err(format!(
                            "checkpoint index does not map weight {name} to shard {path}"
                        )
                        .into());
                    }
                }
                if files
                    .tensors
                    .insert(name.clone(), (shard, info.clone()))
                    .is_some()
                {
                    return Err(format!("duplicate checkpoint weight {name}").into());
                }
            }
            files.shards.push(Shard {
                file,
                payload_start,
            });
        }
        if let Some(index) = &index {
            for name in index.weight_map.keys() {
                if !files.tensors.contains_key(name) {
                    return Err(format!(
                        "checkpoint index weight {name} is missing from its shard"
                    )
                    .into());
                }
            }
        }
        Ok(files)
    }
}

fn checkpoint_file(path: &Path) -> Result<(std::fs::File, safetensors::tensor::Metadata, u64)> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype;
    use std::io::Cursor;

    #[test]
    fn direct_half_conversion_matches_every_bit_pattern() {
        let bytes: Vec<_> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        for dtype in [Dtype::F16, Dtype::BF16] {
            let values = read_weight(
                "weight",
                dtype,
                bytes.len(),
                &mut Cursor::new(&bytes),
                &mut Vec::new(),
                &mut LoadTiming::default(),
            )
            .unwrap();
            for (bits, actual) in (0..=u16::MAX).zip(values.iter()) {
                let expected = match dtype {
                    Dtype::F16 => half::f16::from_bits(bits).to_f32(),
                    _ => half::bf16::from_bits(bits).to_f32(),
                };
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{dtype:?}: {bits:04x}"
                );
            }
        }
    }

    #[test]
    fn direct_weights_cross_chunk_boundaries_and_reuse_bounded_scratch() {
        let mut scratch = Vec::new();
        for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            let width = if dtype == Dtype::F32 { 4 } else { 2 };
            let count = READ_CHUNK_BYTES / width + 13;
            let bytes: Vec<_> = (0..count)
                .flat_map(|i| {
                    let bits = if dtype == Dtype::F32 {
                        // Include negative zero, infinity and NaN payloads unchanged.
                        [
                            0u32,
                            0x8000_0000,
                            0x7f80_0000,
                            0x7fa1_2345,
                            0xffc5_4321,
                            0x3fc0_0000,
                        ][i % 6]
                    } else {
                        u32::from(i as u16)
                    };
                    bits.to_le_bytes().into_iter().take(width)
                })
                .collect();
            let mut reader = Cursor::new(&bytes);
            let values = read_weight(
                "weight",
                dtype,
                bytes.len(),
                &mut reader,
                &mut scratch,
                &mut LoadTiming::default(),
            )
            .unwrap();
            assert_eq!(reader.position() as usize, bytes.len());
            assert_eq!(scratch.len(), READ_CHUNK_BYTES);
            assert_eq!(values.len(), count);
            for (actual, bytes) in values.iter().zip(bytes.chunks_exact(width)) {
                let expected = match dtype {
                    Dtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
                    Dtype::F16 => {
                        half::f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
                    }
                    _ => half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))
                        .to_f32(),
                };
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn direct_weights_reject_bad_types_lengths_and_partial_reads() {
        for (dtype, bytes, expected) in [
            (
                Dtype::I32,
                vec![0; 4],
                "unsupported checkpoint weight dtype",
            ),
            (
                Dtype::BF16,
                vec![0; 3],
                "invalid checkpoint weight byte length",
            ),
        ] {
            let error = read_weight(
                "weight",
                dtype,
                bytes.len(),
                &mut Cursor::new(bytes),
                &mut Vec::new(),
                &mut LoadTiming::default(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected));
        }
        let bytes = vec![0; READ_CHUNK_BYTES];
        assert!(read_weight(
            "weight",
            Dtype::BF16,
            bytes.len() + 2,
            &mut Cursor::new(bytes),
            &mut Vec::new(),
            &mut LoadTiming::default()
        )
        .is_err());
        assert!(read_weight(
            "empty",
            Dtype::F32,
            0,
            &mut Cursor::new([]),
            &mut Vec::new(),
            &mut LoadTiming::default()
        )
        .unwrap()
        .is_empty());
    }
}
