use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::models::autoregressive::{self, AutoregressiveDecoder};
use crate::models::safetensors::{
    parse_safetensors, read_safetensors_file, tensor_f32 as safetensor_f32, SafeTensorLoadError,
};
use crate::runtime::thread_pool::ThreadPool;

use super::{
    Gpt2AssetPaths, Gpt2BackendConfig, Gpt2Error, Gpt2GenerationConfig, Gpt2RustConfig, Result,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: usize,
    pub layer_norm_epsilon: f32,
}

impl Gpt2Config {
    pub fn new(
        vocab_size: usize,
        n_positions: usize,
        n_embd: usize,
        n_layer: usize,
        n_head: usize,
    ) -> Self {
        Self {
            vocab_size,
            n_positions,
            n_embd,
            n_layer,
            n_head,
            n_inner: 4 * n_embd,
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(Gpt2Error::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.n_positions == 0 {
            return Err(Gpt2Error::InvalidConfig(
                "n_positions must be > 0".to_string(),
            ));
        }
        if self.n_embd == 0 {
            return Err(Gpt2Error::InvalidConfig("n_embd must be > 0".to_string()));
        }
        if self.n_layer == 0 {
            return Err(Gpt2Error::InvalidConfig("n_layer must be > 0".to_string()));
        }
        if self.n_head == 0 {
            return Err(Gpt2Error::InvalidConfig("n_head must be > 0".to_string()));
        }
        if self.n_inner == 0 {
            return Err(Gpt2Error::InvalidConfig("n_inner must be > 0".to_string()));
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(Gpt2Error::InvalidConfig(format!(
                "n_embd {} must be divisible by n_head {}",
                self.n_embd, self.n_head
            )));
        }
        if self.layer_norm_epsilon <= 0.0 {
            return Err(Gpt2Error::InvalidConfig(
                "layer_norm_epsilon must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2Weights {
    pub wte: Vec<f32>,
    pub wpe: Vec<f32>,
    pub blocks: Vec<Gpt2BlockWeights>,
    pub ln_f_g: Vec<f32>,
    pub ln_f_b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2BlockWeights {
    pub ln_1_g: Vec<f32>,
    pub ln_1_b: Vec<f32>,
    pub c_attn_w: Vec<f32>,
    pub c_attn_b: Vec<f32>,
    pub c_proj_w: Vec<f32>,
    pub c_proj_b: Vec<f32>,
    pub ln_2_g: Vec<f32>,
    pub ln_2_b: Vec<f32>,
    pub c_fc_w: Vec<f32>,
    pub c_fc_b: Vec<f32>,
    pub c_proj_mlp_w: Vec<f32>,
    pub c_proj_mlp_b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
struct Gpt2QuantizedWeights {
    wte: QuantizedRows,
    blocks: Vec<Gpt2QuantizedBlockWeights>,
}

#[derive(Clone, Debug, PartialEq)]
struct Gpt2QuantizedBlockWeights {
    c_attn_w: QuantizedRows,
    c_proj_w: QuantizedRows,
    c_fc_w: QuantizedRows,
    c_proj_mlp_w: QuantizedRows,
}

#[derive(Clone, Debug, PartialEq)]
struct QuantizedRows {
    values: Vec<i8>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2Model {
    pub config: Gpt2Config,
    pub weights: Arc<Gpt2Weights>,
    pub rust_config: Gpt2RustConfig,
    quantized_weights: Option<Arc<Gpt2QuantizedWeights>>,
    thread_pool: ThreadPool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2Output {
    pub logits: Vec<f32>,
    pub seq_len: usize,
    pub vocab_size: usize,
}

impl Gpt2Output {
    pub fn logits_for_position(&self, pos: usize) -> Result<&[f32]> {
        if pos >= self.seq_len {
            return Err(Gpt2Error::InvalidInput(format!(
                "position {pos} out of range for seq_len {}",
                self.seq_len
            )));
        }
        let start = pos * self.vocab_size;
        Ok(&self.logits[start..start + self.vocab_size])
    }

    pub fn last_logits(&self) -> Result<&[f32]> {
        if self.seq_len == 0 {
            return Err(Gpt2Error::InvalidInput("empty sequence".to_string()));
        }
        self.logits_for_position(self.seq_len - 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2KvCache {
    pub layers: Vec<Gpt2LayerKvCache>,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_head: usize,
    pub head_dim: usize,
}

impl Gpt2KvCache {
    pub fn new(config: &Gpt2Config) -> Result<Self> {
        config.validate()?;
        let head_dim = config.head_dim();
        let layers = (0..config.n_layer)
            .map(|_| Gpt2LayerKvCache::new(config.n_positions, config.n_head, head_dim))
            .collect();
        Ok(Self {
            layers,
            seq_len: 0,
            max_seq_len: config.n_positions,
            n_head: config.n_head,
            head_dim,
        })
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    fn check_compatible(&self, config: &Gpt2Config) -> Result<()> {
        if self.layers.len() != config.n_layer {
            return Err(Gpt2Error::InvalidInput(format!(
                "cache layer count {} does not match n_layer {}",
                self.layers.len(),
                config.n_layer
            )));
        }
        if self.max_seq_len != config.n_positions {
            return Err(Gpt2Error::InvalidInput(format!(
                "cache max_seq_len {} does not match n_positions {}",
                self.max_seq_len, config.n_positions
            )));
        }
        if self.n_head != config.n_head {
            return Err(Gpt2Error::InvalidInput(format!(
                "cache n_head {} does not match n_head {}",
                self.n_head, config.n_head
            )));
        }
        if self.head_dim != config.head_dim() {
            return Err(Gpt2Error::InvalidInput(format!(
                "cache head_dim {} does not match head_dim {}",
                self.head_dim,
                config.head_dim()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2LayerKvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub max_seq_len: usize,
}

impl Gpt2LayerKvCache {
    fn new(max_seq_len: usize, n_head: usize, head_dim: usize) -> Self {
        let len = max_seq_len * n_head * head_dim;
        Self {
            keys: vec![0.0; len],
            values: vec![0.0; len],
            max_seq_len,
        }
    }
}

#[derive(Clone)]
pub struct Gpt2Tokenizer {
    tokenizer: Tokenizer,
}

impl Gpt2Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref()).map_err(|err| {
            Gpt2Error::Asset(format!(
                "failed to load tokenizer {}: {err}",
                path.as_ref().display()
            ))
        })?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| Gpt2Error::Asset(format!("failed to encode prompt: {err}")))?;
        Ok(encoding.get_ids().iter().map(|id| *id as usize).collect())
    }

    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let token_ids: std::result::Result<Vec<u32>, _> =
            token_ids.iter().map(|id| u32::try_from(*id)).collect();
        let token_ids = token_ids
            .map_err(|_| Gpt2Error::InvalidInput("token id does not fit in u32".to_string()))?;
        self.tokenizer
            .decode(&token_ids, true)
            .map_err(|err| Gpt2Error::Asset(format!("failed to decode tokens: {err}")))
    }
}

#[derive(Clone)]
pub struct Gpt2Runtime {
    pub model: Gpt2Model,
    pub tokenizer: Gpt2Tokenizer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokenize_time: Duration,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub total_generation_time: Duration,
    pub first_token_time: Option<Duration>,
    pub operation_profile: Gpt2OperationProfile,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gpt2OperationProfile {
    pub tokenization: Duration,
    pub layer_norm: Duration,
    pub qkv_projection: Duration,
    pub attention: Duration,
    pub attention_projection: Duration,
    pub mlp_fc_projection: Duration,
    pub mlp_projection: Duration,
    pub final_logits: Duration,
    pub decoding: Duration,
}

impl Gpt2GenerationStats {
    pub fn total_model_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    pub fn prefill_tokens_per_second(&self) -> f64 {
        let seconds = self.prefill_time.as_secs_f64();
        if seconds == 0.0 {
            return 0.0;
        }
        self.prompt_tokens as f64 / seconds
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        let seconds = self.decode_time.as_secs_f64();
        if seconds == 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / seconds
    }

    pub fn total_tokens_per_second(&self) -> f64 {
        let seconds = self.total_generation_time.as_secs_f64();
        if seconds == 0.0 {
            return 0.0;
        }
        self.total_model_tokens() as f64 / seconds
    }

    pub fn average_decode_token_time(&self) -> Option<Duration> {
        if self.generated_tokens == 0 {
            return None;
        }
        Some(Duration::from_secs_f64(
            self.decode_time.as_secs_f64() / self.generated_tokens as f64,
        ))
    }
}

impl Gpt2Runtime {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with_backend(model_dir, Gpt2BackendConfig::rust(1)?)
    }

    pub fn from_dir_with_backend(
        model_dir: impl AsRef<Path>,
        backend: Gpt2BackendConfig,
    ) -> Result<Self> {
        let paths = Gpt2AssetPaths::new(model_dir.as_ref());
        let config = load_config(&paths.config)?;
        let weights = load_weights(&paths.weights, &config)?;
        let model = match backend {
            Gpt2BackendConfig::Rust(rust_config) => {
                Gpt2Model::new_with_rust_config(config, weights, rust_config)?
            }
        };
        let tokenizer = Gpt2Tokenizer::from_file(&paths.tokenizer)?;
        Ok(Self { model, tokenizer })
    }

    pub fn generate_greedy_text(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self
            .model
            .generate_greedy_cached(&input_ids, max_new_tokens)?;
        self.tokenizer.decode(&output_ids)
    }

    pub fn stream_greedy_text<F, E>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        mut on_text: F,
    ) -> std::result::Result<String, E>
    where
        F: FnMut(&str) -> std::result::Result<(), E>,
        E: From<Gpt2Error>,
    {
        let input_ids = self.tokenizer.encode(prompt)?;
        let mut output_ids = input_ids.clone();
        let mut decoded = self.tokenizer.decode(&output_ids)?;

        on_text(&decoded)?;

        self.model
            .stream_greedy_tokens::<_, E>(&input_ids, max_new_tokens, |token_id| {
                output_ids.push(token_id);
                self.stream_decoded_token(&output_ids, &mut decoded, token_id, &mut on_text)?;
                Ok(())
            })?;

        Ok(decoded)
    }

    pub fn stream_greedy_text_with_stats<F, E>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        on_text: F,
    ) -> std::result::Result<(String, Gpt2GenerationStats), E>
    where
        F: FnMut(&str) -> std::result::Result<(), E>,
        E: From<Gpt2Error>,
    {
        let generation = Gpt2GenerationConfig::new(max_new_tokens);
        self.stream_text_with_stats(prompt, &generation, on_text)
    }

    pub fn stream_text_with_stats<F, E>(
        &self,
        prompt: &str,
        generation: &Gpt2GenerationConfig,
        mut on_text: F,
    ) -> std::result::Result<(String, Gpt2GenerationStats), E>
    where
        F: FnMut(&str) -> std::result::Result<(), E>,
        E: From<Gpt2Error>,
    {
        generation.validate()?;
        if let Some(eos_token_id) = generation.eos_token_id {
            if eos_token_id >= self.model.config.vocab_size {
                return Err(Gpt2Error::InvalidConfig(format!(
                    "generation eos_token_id {eos_token_id} is outside vocab_size {}",
                    self.model.config.vocab_size
                ))
                .into());
            }
        }

        let mut operation_profile = Gpt2OperationProfile::default();

        let tokenize_start = Instant::now();
        let input_ids = self.tokenizer.encode(prompt)?;
        let tokenize_time = tokenize_start.elapsed();
        operation_profile.tokenization += tokenize_time;

        let mut output_ids = input_ids.clone();
        let decode_text_start = Instant::now();
        let mut decoded = self.tokenizer.decode(&output_ids)?;
        operation_profile.decoding += decode_text_start.elapsed();
        on_text(&decoded)?;

        let mut cache = self.model.new_kv_cache()?;
        let first_token_start = Instant::now();
        let prefill_start = Instant::now();
        let mut logits =
            self.model
                .prefill_profiled(&input_ids, &mut cache, &mut operation_profile)?;
        let prefill_time = prefill_start.elapsed();

        let decode_start = Instant::now();
        let mut first_token_time = None;
        let mut generated_tokens = 0;
        let mut rng = SmallRng::new(generation.seed);

        for _ in 0..generation.max_new_tokens {
            if output_ids.len() >= self.model.config.n_positions {
                break;
            }

            let token_id = select_next_token(&logits, &output_ids, generation, &mut rng)?;
            if generation.eos_token_id == Some(token_id) {
                break;
            }

            output_ids.push(token_id);
            self.stream_decoded_token_profiled(
                &output_ids,
                &mut decoded,
                token_id,
                &mut operation_profile,
                &mut on_text,
            )?;
            generated_tokens += 1;
            if first_token_time.is_none() {
                first_token_time = Some(first_token_start.elapsed());
            }
            let needs_next_logits = generated_tokens < generation.max_new_tokens
                && output_ids.len() < self.model.config.n_positions;
            if needs_next_logits {
                logits = self.model.forward_one_profiled(
                    token_id,
                    &mut cache,
                    &mut operation_profile,
                )?;
            }
        }

        let decode_time = decode_start.elapsed();
        let stats = Gpt2GenerationStats {
            prompt_tokens: input_ids.len(),
            generated_tokens,
            tokenize_time,
            prefill_time,
            decode_time,
            total_generation_time: prefill_time + decode_time,
            first_token_time,
            operation_profile,
        };

        Ok((decoded, stats))
    }

    fn stream_decoded_token<F, E>(
        &self,
        output_ids: &[usize],
        decoded: &mut String,
        token_id: usize,
        on_text: &mut F,
    ) -> std::result::Result<(), E>
    where
        F: FnMut(&str) -> std::result::Result<(), E>,
        E: From<Gpt2Error>,
    {
        let token_text = self.tokenizer.decode(&[token_id])?;
        if is_incremental_decode_safe(&token_text) {
            on_text(&token_text)?;
            decoded.push_str(&token_text);
            return Ok(());
        }

        let next_decoded = self.tokenizer.decode(output_ids)?;
        if let Some(delta) = next_decoded.strip_prefix(decoded.as_str()) {
            on_text(delta)?;
        } else {
            on_text(&token_text)?;
        }
        *decoded = next_decoded;
        Ok(())
    }

    fn stream_decoded_token_profiled<F, E>(
        &self,
        output_ids: &[usize],
        decoded: &mut String,
        token_id: usize,
        profile: &mut Gpt2OperationProfile,
        on_text: &mut F,
    ) -> std::result::Result<(), E>
    where
        F: FnMut(&str) -> std::result::Result<(), E>,
        E: From<Gpt2Error>,
    {
        let decode_text_start = Instant::now();
        let token_text = self.tokenizer.decode(&[token_id])?;
        profile.decoding += decode_text_start.elapsed();
        if is_incremental_decode_safe(&token_text) {
            on_text(&token_text)?;
            decoded.push_str(&token_text);
            return Ok(());
        }

        let decode_text_start = Instant::now();
        let next_decoded = self.tokenizer.decode(output_ids)?;
        profile.decoding += decode_text_start.elapsed();
        if let Some(delta) = next_decoded.strip_prefix(decoded.as_str()) {
            on_text(delta)?;
        } else {
            on_text(&token_text)?;
        }
        *decoded = next_decoded;
        Ok(())
    }
}

fn is_incremental_decode_safe(token_text: &str) -> bool {
    !token_text.is_empty() && !token_text.contains(char::REPLACEMENT_CHARACTER)
}

#[derive(Debug)]
struct Gpt2Scratch {
    x: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    attn_proj: Vec<f32>,
    residual: Vec<f32>,
    mlp: Vec<f32>,
    mlp_proj: Vec<f32>,
    logits: Vec<f32>,
}

impl Gpt2Scratch {
    fn new(config: &Gpt2Config) -> Self {
        Self {
            x: Vec::with_capacity(config.n_embd),
            norm: Vec::with_capacity(config.n_embd),
            qkv: Vec::with_capacity(3 * config.n_embd),
            attn: Vec::with_capacity(config.n_embd),
            attn_proj: Vec::with_capacity(config.n_embd),
            residual: Vec::with_capacity(config.n_embd),
            mlp: Vec::with_capacity(config.n_inner),
            mlp_proj: Vec::with_capacity(config.n_embd),
            logits: Vec::with_capacity(config.vocab_size),
        }
    }
}

#[derive(Debug, Deserialize)]
struct HfGpt2Config {
    vocab_size: usize,
    n_positions: Option<usize>,
    n_ctx: Option<usize>,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_inner: Option<usize>,
    layer_norm_epsilon: Option<f32>,
}

fn load_config(path: &Path) -> Result<Gpt2Config> {
    let data = fs::read_to_string(path).map_err(|err| {
        Gpt2Error::Asset(format!("failed to read config {}: {err}", path.display()))
    })?;
    let hf: HfGpt2Config = serde_json::from_str(&data).map_err(|err| {
        Gpt2Error::Asset(format!("failed to parse config {}: {err}", path.display()))
    })?;
    let n_positions = hf.n_positions.or(hf.n_ctx).ok_or_else(|| {
        Gpt2Error::InvalidConfig("config must contain n_positions or n_ctx".to_string())
    })?;
    let mut config = Gpt2Config::new(hf.vocab_size, n_positions, hf.n_embd, hf.n_layer, hf.n_head);
    if let Some(n_inner) = hf.n_inner {
        config.n_inner = n_inner;
    }
    if let Some(layer_norm_epsilon) = hf.layer_norm_epsilon {
        config.layer_norm_epsilon = layer_norm_epsilon;
    }
    config.validate()?;
    Ok(config)
}

fn load_weights(path: &Path, cfg: &Gpt2Config) -> Result<Gpt2Weights> {
    let bytes = read_safetensors_file(path).map_err(|err| Gpt2Error::Asset(err.to_string()))?;
    let tensors =
        parse_safetensors(path, &bytes).map_err(|err| Gpt2Error::Asset(err.to_string()))?;

    let mut blocks = Vec::with_capacity(cfg.n_layer);
    for layer in 0..cfg.n_layer {
        let prefix = format!("h.{layer}");
        blocks.push(Gpt2BlockWeights {
            ln_1_g: tensor_f32(&tensors, &format!("{prefix}.ln_1.weight"), &[cfg.n_embd])?,
            ln_1_b: tensor_f32(&tensors, &format!("{prefix}.ln_1.bias"), &[cfg.n_embd])?,
            c_attn_w: tensor_f32(
                &tensors,
                &format!("{prefix}.attn.c_attn.weight"),
                &[cfg.n_embd, 3 * cfg.n_embd],
            )?,
            c_attn_b: tensor_f32(
                &tensors,
                &format!("{prefix}.attn.c_attn.bias"),
                &[3 * cfg.n_embd],
            )?,
            c_proj_w: tensor_f32(
                &tensors,
                &format!("{prefix}.attn.c_proj.weight"),
                &[cfg.n_embd, cfg.n_embd],
            )?,
            c_proj_b: tensor_f32(
                &tensors,
                &format!("{prefix}.attn.c_proj.bias"),
                &[cfg.n_embd],
            )?,
            ln_2_g: tensor_f32(&tensors, &format!("{prefix}.ln_2.weight"), &[cfg.n_embd])?,
            ln_2_b: tensor_f32(&tensors, &format!("{prefix}.ln_2.bias"), &[cfg.n_embd])?,
            c_fc_w: tensor_f32(
                &tensors,
                &format!("{prefix}.mlp.c_fc.weight"),
                &[cfg.n_embd, cfg.n_inner],
            )?,
            c_fc_b: tensor_f32(&tensors, &format!("{prefix}.mlp.c_fc.bias"), &[cfg.n_inner])?,
            c_proj_mlp_w: tensor_f32(
                &tensors,
                &format!("{prefix}.mlp.c_proj.weight"),
                &[cfg.n_inner, cfg.n_embd],
            )?,
            c_proj_mlp_b: tensor_f32(
                &tensors,
                &format!("{prefix}.mlp.c_proj.bias"),
                &[cfg.n_embd],
            )?,
        });
    }

    Ok(Gpt2Weights {
        wte: tensor_f32(&tensors, "wte.weight", &[cfg.vocab_size, cfg.n_embd])?,
        wpe: tensor_f32(&tensors, "wpe.weight", &[cfg.n_positions, cfg.n_embd])?,
        blocks,
        ln_f_g: tensor_f32(&tensors, "ln_f.weight", &[cfg.n_embd])?,
        ln_f_b: tensor_f32(&tensors, "ln_f.bias", &[cfg.n_embd])?,
    })
}

fn tensor_f32(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let prefixed_name = format!("transformer.{name}");
    match safetensor_f32(tensors, name, expected_shape) {
        Ok(values) => Ok(values),
        Err(SafeTensorLoadError::TensorNotFound { .. }) => {
            safetensor_f32(tensors, &prefixed_name, expected_shape).map_err(gpt2_tensor_error)
        }
        Err(err) => Err(gpt2_tensor_error(err)),
    }
}

fn gpt2_tensor_error(err: SafeTensorLoadError) -> Gpt2Error {
    match err {
        SafeTensorLoadError::WrongShape {
            name,
            actual,
            expected,
        } => Gpt2Error::InvalidWeights(format!(
            "tensor {name} shape {actual:?} does not match expected {expected:?}"
        )),
        err => Gpt2Error::Asset(err.to_string()),
    }
}

impl Gpt2Model {
    pub fn new(config: Gpt2Config, weights: Gpt2Weights) -> Result<Self> {
        Self::new_with_rust_config(config, weights, Gpt2RustConfig::default())
    }

    pub fn new_with_rust_config(
        config: Gpt2Config,
        mut weights: Gpt2Weights,
        rust_config: Gpt2RustConfig,
    ) -> Result<Self> {
        config.validate()?;
        rust_config.validate()?;
        validate_weights(&config, &weights)?;
        transpose_dense_weights(&config, &mut weights);
        let quantized_weights = if rust_config.quantized_weights {
            Some(Arc::new(quantize_weights(&config, &weights)))
        } else {
            None
        };
        let thread_pool = ThreadPool::new(rust_config.threads);
        Ok(Self {
            config,
            weights: Arc::new(weights),
            rust_config,
            quantized_weights,
            thread_pool,
        })
    }

    fn thread_pool(&self) -> &ThreadPool {
        &self.thread_pool
    }

    pub fn forward(&self, input_ids: &[usize]) -> Result<Gpt2Output> {
        self.validate_input(input_ids)?;

        let cfg = &self.config;
        let t = input_ids.len();
        let c = cfg.n_embd;
        let mut x = vec![0.0f32; t * c];

        for (pos, token_id) in input_ids.iter().copied().enumerate() {
            let tok = row(&self.weights.wte, token_id, c);
            let pos_emb = row(&self.weights.wpe, pos, c);
            let dst = row_mut(&mut x, pos, c);
            for i in 0..c {
                dst[i] = tok[i] + pos_emb[i];
            }
        }

        for layer in 0..self.weights.blocks.len() {
            x = self.block_forward(&x, t, layer);
        }

        layer_norm_in_place(
            &mut x,
            t,
            c,
            &self.weights.ln_f_g,
            &self.weights.ln_f_b,
            cfg.layer_norm_epsilon,
        );

        let mut logits = vec![0.0f32; t * cfg.vocab_size];
        let pool = self.thread_pool();
        for pos in 0..t {
            let hidden = row(&x, pos, c);
            let out = row_mut(&mut logits, pos, cfg.vocab_size);
            out.copy_from_slice(&logits_from_hidden(
                hidden,
                cfg.vocab_size,
                c,
                &self.weights,
                self.quantized_weights.as_deref(),
                pool,
                &self.rust_config,
            ));
        }

        Ok(Gpt2Output {
            logits,
            seq_len: t,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn generate_greedy(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.validate_input(input_ids)?;
        let mut tokens = input_ids.to_vec();
        for _ in 0..max_new_tokens {
            if tokens.len() >= self.config.n_positions {
                break;
            }
            let output = self.forward(&tokens)?;
            let next = argmax(output.last_logits()?);
            tokens.push(next);
        }
        Ok(tokens)
    }

    pub fn new_kv_cache(&self) -> Result<Gpt2KvCache> {
        Gpt2KvCache::new(&self.config)
    }

    pub fn prefill(&self, input_ids: &[usize], cache: &mut Gpt2KvCache) -> Result<Vec<f32>> {
        self.prefill_profiled(input_ids, cache, &mut Gpt2OperationProfile::default())
    }

    fn prefill_profiled(
        &self,
        input_ids: &[usize],
        cache: &mut Gpt2KvCache,
        profile: &mut Gpt2OperationProfile,
    ) -> Result<Vec<f32>> {
        self.validate_input(input_ids)?;
        cache.check_compatible(&self.config)?;
        cache.clear();

        let mut logits = Vec::new();
        let mut scratch = Gpt2Scratch::new(&self.config);
        for token_id in input_ids {
            logits =
                self.forward_one_profiled_with_scratch(*token_id, cache, profile, &mut scratch)?;
        }
        Ok(logits)
    }

    pub fn forward_one(&self, token_id: usize, cache: &mut Gpt2KvCache) -> Result<Vec<f32>> {
        self.forward_one_profiled(token_id, cache, &mut Gpt2OperationProfile::default())
    }

    fn forward_one_profiled(
        &self,
        token_id: usize,
        cache: &mut Gpt2KvCache,
        profile: &mut Gpt2OperationProfile,
    ) -> Result<Vec<f32>> {
        let mut scratch = Gpt2Scratch::new(&self.config);
        self.forward_one_profiled_with_scratch(token_id, cache, profile, &mut scratch)
    }

    fn forward_one_profiled_with_scratch(
        &self,
        token_id: usize,
        cache: &mut Gpt2KvCache,
        profile: &mut Gpt2OperationProfile,
        scratch: &mut Gpt2Scratch,
    ) -> Result<Vec<f32>> {
        cache.check_compatible(&self.config)?;
        if token_id >= self.config.vocab_size {
            return Err(Gpt2Error::InvalidInput(format!(
                "token id {token_id} exceeds vocab_size {}",
                self.config.vocab_size
            )));
        }
        if cache.seq_len >= cache.max_seq_len {
            return Err(Gpt2Error::InvalidInput(format!(
                "cache is full at seq_len {}",
                cache.seq_len
            )));
        }

        let cfg = &self.config;
        let c = cfg.n_embd;
        let pos = cache.seq_len;
        scratch.x.clear();
        scratch.x.resize(c, 0.0);

        let tok = row(&self.weights.wte, token_id, c);
        let pos_emb = row(&self.weights.wpe, pos, c);
        for i in 0..c {
            scratch.x[i] = tok[i] + pos_emb[i];
        }

        for layer in 0..self.weights.blocks.len() {
            self.block_forward_one_profiled(layer, cache, profile, scratch)?;
            std::mem::swap(&mut scratch.x, &mut scratch.residual);
        }

        let start = Instant::now();
        layer_norm_in_place(
            &mut scratch.x,
            1,
            c,
            &self.weights.ln_f_g,
            &self.weights.ln_f_b,
            cfg.layer_norm_epsilon,
        );
        profile.layer_norm += start.elapsed();

        let start = Instant::now();
        logits_from_hidden_into(
            &scratch.x,
            cfg.vocab_size,
            c,
            &self.weights,
            self.quantized_weights.as_deref(),
            self.thread_pool(),
            &self.rust_config,
            &mut scratch.logits,
        );
        profile.final_logits += start.elapsed();
        cache.seq_len += 1;
        Ok(scratch.logits.clone())
    }

    pub fn generate_greedy_cached(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.stream_greedy_tokens(input_ids, max_new_tokens, |_| Ok::<(), Gpt2Error>(()))
    }

    pub fn stream_greedy_tokens<F, E>(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        mut on_token: F,
    ) -> std::result::Result<Vec<usize>, E>
    where
        F: FnMut(usize) -> std::result::Result<(), E>,
        E: From<Gpt2Error>,
    {
        autoregressive::generate(self, input_ids, max_new_tokens, &mut on_token)
    }

    fn validate_input(&self, input_ids: &[usize]) -> Result<()> {
        if input_ids.is_empty() {
            return Err(Gpt2Error::InvalidInput(
                "input_ids must not be empty".to_string(),
            ));
        }
        if input_ids.len() > self.config.n_positions {
            return Err(Gpt2Error::InvalidInput(format!(
                "sequence length {} exceeds n_positions {}",
                input_ids.len(),
                self.config.n_positions
            )));
        }
        for (i, token_id) in input_ids.iter().copied().enumerate() {
            if token_id >= self.config.vocab_size {
                return Err(Gpt2Error::InvalidInput(format!(
                    "token id {token_id} at index {i} exceeds vocab_size {}",
                    self.config.vocab_size
                )));
            }
        }
        Ok(())
    }

    fn block_forward(&self, x: &[f32], t: usize, layer: usize) -> Vec<f32> {
        let cfg = &self.config;
        let c = cfg.n_embd;
        let weights = &self.weights;
        let block = &weights.blocks[layer];
        let mut norm = x.to_vec();
        layer_norm_in_place(
            &mut norm,
            t,
            c,
            &block.ln_1_g,
            &block.ln_1_b,
            cfg.layer_norm_epsilon,
        );

        let qkv = linear_block(
            &norm,
            LinearShape::new(t, c, 3 * c),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::CAttn,
            self.thread_pool(),
            &self.rust_config,
        );
        let attn = causal_self_attention(
            &qkv,
            t,
            cfg.n_head,
            cfg.head_dim(),
            self.thread_pool(),
            &self.rust_config,
        );
        let attn_proj = linear_block(
            &attn,
            LinearShape::new(t, c, c),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::AttnProj,
            self.thread_pool(),
            &self.rust_config,
        );

        let mut residual = x.to_vec();
        add_in_place(&mut residual, &attn_proj);

        let mut norm = residual.clone();
        layer_norm_in_place(
            &mut norm,
            t,
            c,
            &block.ln_2_g,
            &block.ln_2_b,
            cfg.layer_norm_epsilon,
        );

        let mut mlp = linear_block(
            &norm,
            LinearShape::new(t, c, cfg.n_inner),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::MlpFc,
            self.thread_pool(),
            &self.rust_config,
        );
        gelu_in_place(&mut mlp);
        let mlp_proj = linear_block(
            &mlp,
            LinearShape::new(t, cfg.n_inner, c),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::MlpProj,
            self.thread_pool(),
            &self.rust_config,
        );

        add_in_place(&mut residual, &mlp_proj);
        residual
    }

    fn block_forward_one_profiled(
        &self,
        layer: usize,
        cache: &mut Gpt2KvCache,
        profile: &mut Gpt2OperationProfile,
        scratch: &mut Gpt2Scratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let c = cfg.n_embd;
        let weights = &self.weights;
        let block = &weights.blocks[layer];

        scratch.norm.clear();
        scratch.norm.extend_from_slice(&scratch.x);
        let start = Instant::now();
        layer_norm_in_place(
            &mut scratch.norm,
            1,
            c,
            &block.ln_1_g,
            &block.ln_1_b,
            cfg.layer_norm_epsilon,
        );
        profile.layer_norm += start.elapsed();

        let start = Instant::now();
        linear_block_into(
            &scratch.norm,
            LinearShape::new(1, c, 3 * c),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::CAttn,
            self.thread_pool(),
            &self.rust_config,
            &mut scratch.qkv,
        );
        profile.qkv_projection += start.elapsed();

        let start = Instant::now();
        cached_self_attention_into(
            &scratch.qkv,
            cache.seq_len,
            &mut cache.layers[layer],
            cfg.n_head,
            cfg.head_dim(),
            self.thread_pool(),
            &self.rust_config,
            &mut scratch.attn,
        );
        profile.attention += start.elapsed();

        let start = Instant::now();
        linear_block_into(
            &scratch.attn,
            LinearShape::new(1, c, c),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::AttnProj,
            self.thread_pool(),
            &self.rust_config,
            &mut scratch.attn_proj,
        );
        profile.attention_projection += start.elapsed();

        scratch.residual.clear();
        scratch.residual.extend_from_slice(&scratch.x);
        add_in_place(&mut scratch.residual, &scratch.attn_proj);

        scratch.norm.clear();
        scratch.norm.extend_from_slice(&scratch.residual);
        let start = Instant::now();
        layer_norm_in_place(
            &mut scratch.norm,
            1,
            c,
            &block.ln_2_g,
            &block.ln_2_b,
            cfg.layer_norm_epsilon,
        );
        profile.layer_norm += start.elapsed();

        let start = Instant::now();
        linear_block_into(
            &scratch.norm,
            LinearShape::new(1, c, cfg.n_inner),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::MlpFc,
            self.thread_pool(),
            &self.rust_config,
            &mut scratch.mlp,
        );
        profile.mlp_fc_projection += start.elapsed();
        gelu_in_place(&mut scratch.mlp);

        let start = Instant::now();
        linear_block_into(
            &scratch.mlp,
            LinearShape::new(1, cfg.n_inner, c),
            weights,
            self.quantized_weights.as_deref(),
            layer,
            BlockLinear::MlpProj,
            self.thread_pool(),
            &self.rust_config,
            &mut scratch.mlp_proj,
        );
        profile.mlp_projection += start.elapsed();

        add_in_place(&mut scratch.residual, &scratch.mlp_proj);
        Ok(())
    }
}

impl AutoregressiveDecoder for Gpt2Model {
    type Cache = Gpt2KvCache;
    type Logits = Vec<f32>;
    type Error = Gpt2Error;

    fn max_context_len(&self) -> usize {
        self.config.n_positions
    }

    fn new_cache(&self) -> Result<Self::Cache> {
        self.new_kv_cache()
    }

    fn prefill(&self, input_ids: &[usize], cache: &mut Self::Cache) -> Result<Self::Logits> {
        Gpt2Model::prefill(self, input_ids, cache)
    }

    fn forward_one(&self, token_id: usize, cache: &mut Self::Cache) -> Result<Self::Logits> {
        Gpt2Model::forward_one(self, token_id, cache)
    }

    fn select_next_token(&self, logits: &Self::Logits) -> Result<usize> {
        Ok(argmax(logits))
    }
}

fn validate_weights(cfg: &Gpt2Config, w: &Gpt2Weights) -> Result<()> {
    expect_len("wte", &w.wte, cfg.vocab_size * cfg.n_embd)?;
    expect_len("wpe", &w.wpe, cfg.n_positions * cfg.n_embd)?;
    expect_len("ln_f_g", &w.ln_f_g, cfg.n_embd)?;
    expect_len("ln_f_b", &w.ln_f_b, cfg.n_embd)?;
    if w.blocks.len() != cfg.n_layer {
        return Err(Gpt2Error::InvalidWeights(format!(
            "blocks len {} does not match n_layer {}",
            w.blocks.len(),
            cfg.n_layer
        )));
    }

    for (i, block) in w.blocks.iter().enumerate() {
        let prefix = format!("blocks[{i}]");
        expect_len(&format!("{prefix}.ln_1_g"), &block.ln_1_g, cfg.n_embd)?;
        expect_len(&format!("{prefix}.ln_1_b"), &block.ln_1_b, cfg.n_embd)?;
        expect_len(
            &format!("{prefix}.c_attn_w"),
            &block.c_attn_w,
            cfg.n_embd * 3 * cfg.n_embd,
        )?;
        expect_len(
            &format!("{prefix}.c_attn_b"),
            &block.c_attn_b,
            3 * cfg.n_embd,
        )?;
        expect_len(
            &format!("{prefix}.c_proj_w"),
            &block.c_proj_w,
            cfg.n_embd * cfg.n_embd,
        )?;
        expect_len(&format!("{prefix}.c_proj_b"), &block.c_proj_b, cfg.n_embd)?;
        expect_len(&format!("{prefix}.ln_2_g"), &block.ln_2_g, cfg.n_embd)?;
        expect_len(&format!("{prefix}.ln_2_b"), &block.ln_2_b, cfg.n_embd)?;
        expect_len(
            &format!("{prefix}.c_fc_w"),
            &block.c_fc_w,
            cfg.n_embd * cfg.n_inner,
        )?;
        expect_len(&format!("{prefix}.c_fc_b"), &block.c_fc_b, cfg.n_inner)?;
        expect_len(
            &format!("{prefix}.c_proj_mlp_w"),
            &block.c_proj_mlp_w,
            cfg.n_inner * cfg.n_embd,
        )?;
        expect_len(
            &format!("{prefix}.c_proj_mlp_b"),
            &block.c_proj_mlp_b,
            cfg.n_embd,
        )?;
    }

    Ok(())
}

fn expect_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(Gpt2Error::InvalidWeights(format!(
            "{name} len {} does not match expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn transpose_dense_weights(cfg: &Gpt2Config, weights: &mut Gpt2Weights) {
    for block in &mut weights.blocks {
        block.c_attn_w = transpose_in_out(&block.c_attn_w, cfg.n_embd, 3 * cfg.n_embd);
        block.c_proj_w = transpose_in_out(&block.c_proj_w, cfg.n_embd, cfg.n_embd);
        block.c_fc_w = transpose_in_out(&block.c_fc_w, cfg.n_embd, cfg.n_inner);
        block.c_proj_mlp_w = transpose_in_out(&block.c_proj_mlp_w, cfg.n_inner, cfg.n_embd);
    }
}

fn transpose_in_out(weight: &[f32], in_features: usize, out_features: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for i in 0..in_features {
        for o in 0..out_features {
            transposed[o * in_features + i] = weight[i * out_features + o];
        }
    }
    transposed
}

fn quantize_weights(config: &Gpt2Config, weights: &Gpt2Weights) -> Gpt2QuantizedWeights {
    Gpt2QuantizedWeights {
        wte: QuantizedRows::from_f32(&weights.wte, config.vocab_size, config.n_embd),
        blocks: weights
            .blocks
            .iter()
            .map(|block| Gpt2QuantizedBlockWeights {
                c_attn_w: QuantizedRows::from_f32(
                    &block.c_attn_w,
                    3 * config.n_embd,
                    config.n_embd,
                ),
                c_proj_w: QuantizedRows::from_f32(&block.c_proj_w, config.n_embd, config.n_embd),
                c_fc_w: QuantizedRows::from_f32(&block.c_fc_w, config.n_inner, config.n_embd),
                c_proj_mlp_w: QuantizedRows::from_f32(
                    &block.c_proj_mlp_w,
                    config.n_embd,
                    config.n_inner,
                ),
            })
            .collect(),
    }
}

impl QuantizedRows {
    fn from_f32(values: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut quantized = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(rows);
        for r in 0..rows {
            let src = row(values, r, cols);
            let max_abs = src.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales.push(scale);
            for value in src {
                let q = (value / scale).round().clamp(-127.0, 127.0) as i8;
                quantized.push(q);
            }
        }
        Self {
            values: quantized,
            scales,
            rows,
            cols,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockLinear {
    CAttn,
    AttnProj,
    MlpFc,
    MlpProj,
}

impl BlockLinear {
    fn chunk_size(self, config: &Gpt2RustConfig) -> usize {
        match self {
            BlockLinear::CAttn => config.qkv_chunk_size,
            BlockLinear::AttnProj => config.attention_projection_chunk_size,
            BlockLinear::MlpFc => config.mlp_fc_chunk_size,
            BlockLinear::MlpProj => config.mlp_projection_chunk_size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearShape {
    rows: usize,
    in_features: usize,
    out_features: usize,
}

impl LinearShape {
    fn new(rows: usize, in_features: usize, out_features: usize) -> Self {
        Self {
            rows,
            in_features,
            out_features,
        }
    }

    fn out_len(self) -> usize {
        self.rows * self.out_features
    }

    fn work_items(self) -> usize {
        self.rows * self.in_features * self.out_features
    }
}

fn block_linear_slices(
    weights: &Gpt2Weights,
    layer: usize,
    which: BlockLinear,
) -> (&[f32], &[f32]) {
    // Dense projection matrices are transposed during model construction:
    // weight layout is [out_features, in_features].
    let block = &weights.blocks[layer];
    match which {
        BlockLinear::CAttn => (&block.c_attn_w, &block.c_attn_b),
        BlockLinear::AttnProj => (&block.c_proj_w, &block.c_proj_b),
        BlockLinear::MlpFc => (&block.c_fc_w, &block.c_fc_b),
        BlockLinear::MlpProj => (&block.c_proj_mlp_w, &block.c_proj_mlp_b),
    }
}

fn quantized_block_linear_slices<'a>(
    quantized_weights: &'a Gpt2QuantizedWeights,
    weights: &'a Gpt2Weights,
    layer: usize,
    which: BlockLinear,
) -> (&'a QuantizedRows, &'a [f32]) {
    let block = &quantized_weights.blocks[layer];
    let f32_block = &weights.blocks[layer];
    match which {
        BlockLinear::CAttn => (&block.c_attn_w, &f32_block.c_attn_b),
        BlockLinear::AttnProj => (&block.c_proj_w, &f32_block.c_proj_b),
        BlockLinear::MlpFc => (&block.c_fc_w, &f32_block.c_fc_b),
        BlockLinear::MlpProj => (&block.c_proj_mlp_w, &f32_block.c_proj_mlp_b),
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_block(
    x: &[f32],
    shape: LinearShape,
    weights: &Gpt2Weights,
    quantized_weights: Option<&Gpt2QuantizedWeights>,
    layer: usize,
    which: BlockLinear,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
) -> Vec<f32> {
    let mut out = Vec::new();
    linear_block_into(
        x,
        shape,
        weights,
        quantized_weights,
        layer,
        which,
        pool,
        rust_config,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn linear_block_into(
    x: &[f32],
    shape: LinearShape,
    weights: &Gpt2Weights,
    quantized_weights: Option<&Gpt2QuantizedWeights>,
    layer: usize,
    which: BlockLinear,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
    out: &mut Vec<f32>,
) {
    if let Some(quantized_weights) = quantized_weights {
        let (weight, bias) =
            quantized_block_linear_slices(quantized_weights, weights, layer, which);
        quantized_linear_into(x, shape, weight, bias, pool, rust_config, which, out);
        return;
    }

    let (weight, bias) = block_linear_slices(weights, layer, which);
    if pool.threads() == 1 || shape.work_items() < rust_config.dense_parallel_threshold {
        linear_into(
            x,
            shape.rows,
            shape.in_features,
            weight,
            bias,
            shape.out_features,
            out,
        );
        return;
    }

    let chunks = pool.scoped_parallel_chunks(
        shape.out_len(),
        which.chunk_size(rust_config),
        |start, end| {
            let mut values = Vec::with_capacity(end - start);
            for index in start..end {
                let r = index / shape.out_features;
                let o = index % shape.out_features;
                let src = row(x, r, shape.in_features);
                let weight_row = row(weight, o, shape.in_features);
                values.push(bias[o] + dot(src, weight_row));
            }
            values
        },
    );

    out.clear();
    out.reserve(shape.out_len());
    for mut chunk in chunks {
        out.append(&mut chunk);
    }
}

fn linear_into(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    out_features: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(rows * out_features, 0.0);
    for r in 0..rows {
        let src = row(x, r, in_features);
        let dst = row_mut(out, r, out_features);
        for o in 0..out_features {
            let weight_row = row(weight, o, in_features);
            dst[o] = bias[o] + dot(src, weight_row);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quantized_linear_into(
    x: &[f32],
    shape: LinearShape,
    weight: &QuantizedRows,
    bias: &[f32],
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
    which: BlockLinear,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(weight.rows, shape.out_features);
    debug_assert_eq!(weight.cols, shape.in_features);
    if pool.threads() == 1 || shape.work_items() < rust_config.dense_parallel_threshold {
        out.clear();
        out.resize(shape.out_len(), 0.0);
        for r in 0..shape.rows {
            let src = row(x, r, shape.in_features);
            let dst = row_mut(out, r, shape.out_features);
            for o in 0..shape.out_features {
                dst[o] = bias[o] + quantized_dot(src, weight, o);
            }
        }
        return;
    }

    let chunks = pool.scoped_parallel_chunks(
        shape.out_len(),
        which.chunk_size(rust_config),
        |start, end| {
            let mut values = Vec::with_capacity(end - start);
            for index in start..end {
                let r = index / shape.out_features;
                let o = index % shape.out_features;
                let src = row(x, r, shape.in_features);
                values.push(bias[o] + quantized_dot(src, weight, o));
            }
            values
        },
    );

    out.clear();
    out.reserve(shape.out_len());
    for mut chunk in chunks {
        out.append(&mut chunk);
    }
}

fn causal_self_attention(
    qkv: &[f32],
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
) -> Vec<f32> {
    let n_embd = n_head * head_dim;
    let mut out = vec![0.0f32; seq_len * n_embd];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let work_items = seq_len * seq_len * n_head * head_dim;

    if pool.threads() > 1 && work_items >= rust_config.attention_head_parallel_threshold {
        let heads = pool.scoped_parallel_chunks(n_head, 1, |start, end| {
            let mut head_outputs = Vec::with_capacity(end - start);
            for h in start..end {
                let mut head_out = vec![0.0f32; seq_len * head_dim];
                for q_pos in 0..seq_len {
                    let mut scores = vec![0.0f32; q_pos + 1];
                    for k_pos in 0..=q_pos {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            let q = qkv[q_pos * 3 * n_embd + h * head_dim + d];
                            let k = qkv[k_pos * 3 * n_embd + n_embd + h * head_dim + d];
                            score += q * k;
                        }
                        scores[k_pos] = score * scale;
                    }
                    softmax_in_place(&mut scores);

                    for k_pos in 0..=q_pos {
                        let prob = scores[k_pos];
                        for d in 0..head_dim {
                            let v = qkv[k_pos * 3 * n_embd + 2 * n_embd + h * head_dim + d];
                            head_out[q_pos * head_dim + d] += prob * v;
                        }
                    }
                }
                head_outputs.push((h, head_out));
            }
            head_outputs
        });

        for (h, head_out) in heads.into_iter().flatten() {
            for q_pos in 0..seq_len {
                let src = row(&head_out, q_pos, head_dim);
                let dst_start = q_pos * n_embd + h * head_dim;
                out[dst_start..dst_start + head_dim].copy_from_slice(src);
            }
        }
        return out;
    }

    for h in 0..n_head {
        for q_pos in 0..seq_len {
            let mut scores = vec![0.0f32; q_pos + 1];
            for k_pos in 0..=q_pos {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q = qkv[q_pos * 3 * n_embd + h * head_dim + d];
                    let k = qkv[k_pos * 3 * n_embd + n_embd + h * head_dim + d];
                    score += q * k;
                }
                scores[k_pos] = score * scale;
            }
            softmax_in_place(&mut scores);

            for k_pos in 0..=q_pos {
                let prob = scores[k_pos];
                for d in 0..head_dim {
                    let v = qkv[k_pos * 3 * n_embd + 2 * n_embd + h * head_dim + d];
                    out[q_pos * n_embd + h * head_dim + d] += prob * v;
                }
            }
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn cached_self_attention_into(
    qkv: &[f32],
    pos: usize,
    cache: &mut Gpt2LayerKvCache,
    n_head: usize,
    head_dim: usize,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
    out: &mut Vec<f32>,
) {
    let n_embd = n_head * head_dim;
    out.clear();
    out.resize(n_embd, 0.0);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for h in 0..n_head {
        for d in 0..head_dim {
            let idx = kv_index(pos, h, d, cache.max_seq_len, head_dim);
            cache.keys[idx] = qkv[n_embd + h * head_dim + d];
            cache.values[idx] = qkv[2 * n_embd + h * head_dim + d];
        }
    }

    let work_items = (pos + 1) * n_head * head_dim;
    if pool.threads() > 1 && work_items >= rust_config.attention_head_parallel_threshold {
        let heads = pool.scoped_parallel_chunks(n_head, 1, |start, end| {
            let mut head_outputs = Vec::with_capacity(end - start);
            for h in start..end {
                let mut head_out = vec![0.0f32; head_dim];
                let mut scores = vec![0.0f32; pos + 1];
                for (k_pos, score) in scores.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for d in 0..head_dim {
                        let q = qkv[h * head_dim + d];
                        let k = cache.keys[kv_index(k_pos, h, d, cache.max_seq_len, head_dim)];
                        sum += q * k;
                    }
                    *score = sum * scale;
                }
                softmax_in_place(&mut scores);

                for (k_pos, prob) in scores.iter().copied().enumerate() {
                    for (d, head_value) in head_out.iter_mut().enumerate().take(head_dim) {
                        let v = cache.values[kv_index(k_pos, h, d, cache.max_seq_len, head_dim)];
                        *head_value += prob * v;
                    }
                }
                head_outputs.push((h, head_out));
            }
            head_outputs
        });

        for (h, head_out) in heads.into_iter().flatten() {
            let dst_start = h * head_dim;
            out[dst_start..dst_start + head_dim].copy_from_slice(&head_out);
        }
        return;
    }

    for h in 0..n_head {
        let mut scores = vec![0.0f32; pos + 1];
        for (k_pos, score) in scores.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for d in 0..head_dim {
                let q = qkv[h * head_dim + d];
                let k = cache.keys[kv_index(k_pos, h, d, cache.max_seq_len, head_dim)];
                sum += q * k;
            }
            *score = sum * scale;
        }
        softmax_in_place(&mut scores);

        for (k_pos, prob) in scores.iter().copied().enumerate() {
            for d in 0..head_dim {
                let v = cache.values[kv_index(k_pos, h, d, cache.max_seq_len, head_dim)];
                out[h * head_dim + d] += prob * v;
            }
        }
    }
}

fn logits_from_hidden(
    hidden: &[f32],
    vocab_size: usize,
    n_embd: usize,
    weights: &Gpt2Weights,
    quantized_weights: Option<&Gpt2QuantizedWeights>,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
) -> Vec<f32> {
    let mut logits = Vec::new();
    logits_from_hidden_into(
        hidden,
        vocab_size,
        n_embd,
        weights,
        quantized_weights,
        pool,
        rust_config,
        &mut logits,
    );
    logits
}

#[allow(clippy::too_many_arguments)]
fn logits_from_hidden_into(
    hidden: &[f32],
    vocab_size: usize,
    n_embd: usize,
    weights: &Gpt2Weights,
    quantized_weights: Option<&Gpt2QuantizedWeights>,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
    logits: &mut Vec<f32>,
) {
    if let Some(quantized_weights) = quantized_weights {
        quantized_logits_from_hidden_into(
            hidden,
            vocab_size,
            n_embd,
            &quantized_weights.wte,
            pool,
            rust_config,
            logits,
        );
        return;
    }

    if pool.threads() == 1 {
        logits.clear();
        logits.resize(vocab_size, 0.0);
        for (token, logit) in logits.iter_mut().enumerate() {
            *logit = dot(hidden, row(&weights.wte, token, n_embd));
        }
        return;
    }

    let chunks =
        pool.scoped_parallel_chunks(vocab_size, rust_config.logits_chunk_size, |start, end| {
            let mut values = Vec::with_capacity(end - start);
            for token in start..end {
                values.push(dot(hidden, row(&weights.wte, token, n_embd)));
            }
            values
        });

    logits.clear();
    logits.reserve(vocab_size);
    for mut chunk in chunks {
        logits.append(&mut chunk);
    }
}

fn quantized_logits_from_hidden_into(
    hidden: &[f32],
    vocab_size: usize,
    n_embd: usize,
    weight: &QuantizedRows,
    pool: &ThreadPool,
    rust_config: &Gpt2RustConfig,
    logits: &mut Vec<f32>,
) {
    debug_assert_eq!(weight.rows, vocab_size);
    debug_assert_eq!(weight.cols, n_embd);
    if pool.threads() == 1 {
        logits.clear();
        logits.resize(vocab_size, 0.0);
        for (token, logit) in logits.iter_mut().enumerate() {
            *logit = quantized_dot(hidden, weight, token);
        }
        return;
    }

    let chunks =
        pool.scoped_parallel_chunks(vocab_size, rust_config.logits_chunk_size, |start, end| {
            let mut values = Vec::with_capacity(end - start);
            for token in start..end {
                values.push(quantized_dot(hidden, weight, token));
            }
            values
        });

    logits.clear();
    logits.reserve(vocab_size);
    for mut chunk in chunks {
        logits.append(&mut chunk);
    }
}

fn kv_index(pos: usize, head: usize, dim: usize, max_seq_len: usize, head_dim: usize) -> usize {
    // Per-head contiguous blocks make cached attention scan sequential key/value rows
    // for one head at a time: [head][position][head_dim].
    (head * max_seq_len + pos) * head_dim + dim
}

fn layer_norm_in_place(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    for r in 0..rows {
        let row = row_mut(x, r, cols);
        let mean = row.iter().sum::<f32>() / cols as f32;
        let variance = row
            .iter()
            .map(|v| {
                let delta = *v - mean;
                delta * delta
            })
            .sum::<f32>()
            / cols as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for c in 0..cols {
            row[c] = (row[c] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
}

fn gelu_in_place(values: &mut [f32]) {
    for value in values {
        let x = *value;
        *value = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

fn quantized_dot(a: &[f32], rows: &QuantizedRows, row_index: usize) -> f32 {
    debug_assert_eq!(a.len(), rows.cols);
    debug_assert!(row_index < rows.rows);
    let start = row_index * rows.cols;
    let values = &rows.values[start..start + rows.cols];
    let scale = rows.scales[row_index];
    let mut sum = 0.0f32;
    for (x, q) in a.iter().zip(values.iter()) {
        sum += *x * (*q as f32 * scale);
    }
    sum
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(a, b) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { dot_avx(a, b) };
        }
        dot_scalar(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

#[allow(dead_code)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vld1q_f32, vmlaq_f32};

    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= a.len() {
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        acc = vmlaq_f32(acc, av, bv);
        i += 4;
    }

    let mut sum = vaddvq_f32(acc);
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f32>();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f32>();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn row(values: &[f32], row: usize, cols: usize) -> &[f32] {
    &values[row * cols..(row + 1) * cols]
}

fn row_mut(values: &mut [f32], row: usize, cols: usize) -> &mut [f32] {
    &mut values[row * cols..(row + 1) * cols]
}

#[derive(Clone, Debug)]
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

fn select_next_token(
    logits: &[f32],
    history: &[usize],
    generation: &Gpt2GenerationConfig,
    rng: &mut SmallRng,
) -> Result<usize> {
    let mut adjusted = logits.to_vec();
    apply_repeat_penalty(
        &mut adjusted,
        history,
        generation.repeat_penalty,
        generation.repeat_last_n,
    );

    if generation.temperature <= 0.0 {
        return Ok(argmax(&adjusted));
    }

    sample_logits(&adjusted, generation, rng)
}

fn apply_repeat_penalty(logits: &mut [f32], history: &[usize], penalty: f32, last_n: usize) {
    if penalty == 1.0 || last_n == 0 {
        return;
    }

    let mut seen = Vec::new();
    for token_id in history.iter().rev().take(last_n).copied() {
        if token_id >= logits.len() || seen.contains(&token_id) {
            continue;
        }
        seen.push(token_id);
        if logits[token_id] < 0.0 {
            logits[token_id] *= penalty;
        } else {
            logits[token_id] /= penalty;
        }
    }
}

fn sample_logits(
    logits: &[f32],
    generation: &Gpt2GenerationConfig,
    rng: &mut SmallRng,
) -> Result<usize> {
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .collect();
    if candidates.is_empty() {
        return Err(Gpt2Error::InvalidInput(
            "cannot sample from empty or non-finite logits".to_string(),
        ));
    }

    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if let Some(top_k) = generation.top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }

    let max_logit = candidates[0].1 as f64 / generation.temperature as f64;
    let mut weighted: Vec<(usize, f64)> = candidates
        .into_iter()
        .map(|(token_id, logit)| {
            let scaled = logit as f64 / generation.temperature as f64;
            (token_id, (scaled - max_logit).exp())
        })
        .collect();

    if let Some(top_p) = generation.top_p {
        let full_total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
        if full_total > 0.0 {
            let mut cumulative = 0.0;
            let mut keep_len = 0;
            for (_, weight) in &weighted {
                cumulative += *weight;
                keep_len += 1;
                if cumulative / full_total >= top_p as f64 {
                    break;
                }
            }
            weighted.truncate(keep_len.max(1));
        }
    }

    let total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
    if total <= 0.0 || !total.is_finite() {
        return Ok(weighted[0].0);
    }

    let fallback = weighted.last().map(|(token_id, _)| *token_id).unwrap();
    let mut target = rng.next_unit_f64() * total;
    for (token_id, weight) in weighted {
        target -= weight;
        if target <= 0.0 {
            return Ok(token_id);
        }
    }

    Ok(fallback)
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_weight_shapes() {
        let cfg = tiny_config();
        let mut weights = tiny_weights(&cfg);
        weights.wte.pop();

        let err = Gpt2Model::new(cfg, weights).expect_err("invalid weights should fail");
        assert!(err.to_string().contains("wte len"));
    }

    #[test]
    fn forward_returns_logits_for_each_position() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new(cfg.clone(), tiny_weights(&cfg))?;
        let out = model.forward(&[1, 2, 3])?;

        assert_eq!(out.seq_len, 3);
        assert_eq!(out.vocab_size, cfg.vocab_size);
        assert_eq!(out.logits.len(), 3 * cfg.vocab_size);
        assert!(out.logits.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn greedy_generation_appends_tokens_until_context_limit() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new(cfg.clone(), tiny_weights(&cfg))?;
        let tokens = model.generate_greedy(&[0, 1], 10)?;

        assert_eq!(tokens.len(), cfg.n_positions);
        assert_eq!(&tokens[..2], &[0, 1]);
        assert!(tokens.iter().all(|id| *id < cfg.vocab_size));
        Ok(())
    }

    #[test]
    fn cached_prefill_matches_full_forward_last_logits() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new(cfg.clone(), tiny_weights(&cfg))?;
        let input = [1, 2, 3];

        let full = model.forward(&input)?;
        let mut cache = model.new_kv_cache()?;
        let cached = model.prefill(&input, &mut cache)?;

        assert_eq!(cache.seq_len, input.len());
        assert_close(full.last_logits()?, &cached, 1e-5);
        Ok(())
    }

    #[test]
    fn cached_greedy_generation_matches_full_generation() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new(cfg.clone(), tiny_weights(&cfg))?;

        let full = model.generate_greedy(&[0, 1], 3)?;
        let cached = model.generate_greedy_cached(&[0, 1], 3)?;

        assert_eq!(cached, full);
        Ok(())
    }

    #[test]
    fn streams_generated_tokens_in_order() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new(cfg.clone(), tiny_weights(&cfg))?;
        let mut streamed = Vec::new();

        let tokens = model.stream_greedy_tokens(&[0, 1], 3, |token_id| {
            streamed.push(token_id);
            Ok::<(), Gpt2Error>(())
        })?;

        assert_eq!(&tokens[..2], &[0, 1]);
        assert_eq!(streamed, tokens[2..]);
        Ok(())
    }

    #[test]
    fn implements_generic_autoregressive_decoder() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new(cfg.clone(), tiny_weights(&cfg))?;

        let direct = model.generate_greedy_cached(&[0, 1], 3)?;
        let generic = autoregressive::generate(&model, &[0, 1], 3, |_| Ok::<(), Gpt2Error>(()))?;

        assert_eq!(generic, direct);
        Ok(())
    }

    #[test]
    fn threaded_rust_generation_matches_single_threaded() -> Result<()> {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let single = Gpt2Model::new_with_rust_config(
            cfg.clone(),
            weights.clone(),
            Gpt2RustConfig {
                threads: 1,
                ..Gpt2RustConfig::default()
            },
        )?;
        let threaded = Gpt2Model::new_with_rust_config(
            cfg,
            weights,
            Gpt2RustConfig {
                threads: 3,
                ..Gpt2RustConfig::default()
            },
        )?;

        let single_tokens = single.generate_greedy_cached(&[0, 1], 3)?;
        let threaded_tokens = threaded.generate_greedy_cached(&[0, 1], 3)?;

        assert_eq!(threaded_tokens, single_tokens);
        Ok(())
    }

    #[test]
    fn quantized_rust_forward_returns_finite_logits() -> Result<()> {
        let cfg = tiny_config();
        let model = Gpt2Model::new_with_rust_config(
            cfg.clone(),
            tiny_weights(&cfg),
            Gpt2RustConfig {
                quantized_weights: true,
                ..Gpt2RustConfig::default()
            },
        )?;

        let out = model.forward(&[1, 2, 3])?;

        assert_eq!(out.logits.len(), 3 * cfg.vocab_size);
        assert!(out.logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn generation_stats_report_token_rates() {
        let stats = Gpt2GenerationStats {
            prompt_tokens: 4,
            generated_tokens: 6,
            tokenize_time: Duration::from_millis(5),
            prefill_time: Duration::from_millis(20),
            decode_time: Duration::from_millis(30),
            total_generation_time: Duration::from_millis(50),
            first_token_time: Some(Duration::from_millis(22)),
            operation_profile: Gpt2OperationProfile::default(),
        };

        assert_eq!(stats.total_model_tokens(), 10);
        assert_eq!(stats.prefill_tokens_per_second(), 200.0);
        assert_eq!(stats.decode_tokens_per_second(), 200.0);
        assert_eq!(stats.total_tokens_per_second(), 200.0);
        assert_eq!(
            stats.average_decode_token_time(),
            Some(Duration::from_millis(5))
        );
    }

    #[test]
    fn repeat_penalty_can_change_greedy_selection() -> Result<()> {
        let generation = Gpt2GenerationConfig {
            repeat_penalty: 2.0,
            ..Gpt2GenerationConfig::new(1)
        };
        let mut rng = SmallRng::new(generation.seed);

        let token = select_next_token(&[1.0, 3.0, 2.0], &[1], &generation, &mut rng)?;

        assert_eq!(token, 2);
        Ok(())
    }

    #[test]
    fn top_k_one_samples_best_candidate() -> Result<()> {
        let generation = Gpt2GenerationConfig {
            temperature: 1.0,
            top_k: Some(1),
            ..Gpt2GenerationConfig::new(1)
        };
        let mut rng = SmallRng::new(generation.seed);

        let token = select_next_token(&[0.0, 10.0, 9.0], &[], &generation, &mut rng)?;

        assert_eq!(token, 1);
        Ok(())
    }

    #[test]
    fn incremental_decode_rejects_empty_and_replacement_text() {
        assert!(is_incremental_decode_safe(" hello"));
        assert!(!is_incremental_decode_safe(""));
        assert!(!is_incremental_decode_safe("\u{fffd}"));
        assert!(!is_incremental_decode_safe("a\u{fffd}"));
    }

    #[test]
    fn dot_matches_scalar_reference() {
        let a = patterned(37, 0.013);
        let b = patterned(37, -0.021);

        assert!((dot(&a, &b) - dot_scalar(&a, &b)).abs() <= 1e-6);
    }

    fn tiny_config() -> Gpt2Config {
        Gpt2Config {
            vocab_size: 8,
            n_positions: 5,
            n_embd: 4,
            n_layer: 1,
            n_head: 2,
            n_inner: 8,
            layer_norm_epsilon: 1e-5,
        }
    }

    fn tiny_weights(cfg: &Gpt2Config) -> Gpt2Weights {
        Gpt2Weights {
            wte: patterned(cfg.vocab_size * cfg.n_embd, 0.03),
            wpe: patterned(cfg.n_positions * cfg.n_embd, 0.02),
            blocks: vec![Gpt2BlockWeights {
                ln_1_g: vec![1.0; cfg.n_embd],
                ln_1_b: vec![0.0; cfg.n_embd],
                c_attn_w: patterned(cfg.n_embd * 3 * cfg.n_embd, 0.01),
                c_attn_b: vec![0.0; 3 * cfg.n_embd],
                c_proj_w: patterned(cfg.n_embd * cfg.n_embd, 0.015),
                c_proj_b: vec![0.0; cfg.n_embd],
                ln_2_g: vec![1.0; cfg.n_embd],
                ln_2_b: vec![0.0; cfg.n_embd],
                c_fc_w: patterned(cfg.n_embd * cfg.n_inner, 0.01),
                c_fc_b: vec![0.0; cfg.n_inner],
                c_proj_mlp_w: patterned(cfg.n_inner * cfg.n_embd, 0.012),
                c_proj_mlp_b: vec![0.0; cfg.n_embd],
            }],
            ln_f_g: vec![1.0; cfg.n_embd],
            ln_f_b: vec![0.0; cfg.n_embd],
        }
    }

    fn patterned(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + 11) % 23) as f32 - 11.0) * scale)
            .collect()
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (*l - *r).abs() <= tolerance,
                "values differ at {i}: left={l} right={r}"
            );
        }
    }
}
