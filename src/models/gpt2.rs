use std::error;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use tokenizers::Tokenizer;

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
pub struct Gpt2Model {
    pub config: Gpt2Config,
    pub weights: Gpt2Weights,
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
}

impl Gpt2LayerKvCache {
    fn new(max_seq_len: usize, n_head: usize, head_dim: usize) -> Self {
        let len = max_seq_len * n_head * head_dim;
        Self {
            keys: vec![0.0; len],
            values: vec![0.0; len],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gpt2Error {
    Asset(String),
    InvalidConfig(String),
    InvalidInput(String),
    InvalidWeights(String),
}

impl fmt::Display for Gpt2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gpt2Error::Asset(msg) => write!(f, "GPT-2 asset error: {msg}"),
            Gpt2Error::InvalidConfig(msg) => write!(f, "invalid GPT-2 config: {msg}"),
            Gpt2Error::InvalidInput(msg) => write!(f, "invalid GPT-2 input: {msg}"),
            Gpt2Error::InvalidWeights(msg) => write!(f, "invalid GPT-2 weights: {msg}"),
        }
    }
}

impl error::Error for Gpt2Error {}

pub type Result<T> = std::result::Result<T, Gpt2Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2AssetPaths {
    pub model_dir: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

impl Gpt2AssetPaths {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        Self {
            config: model_dir.join("config.json"),
            tokenizer: model_dir.join("tokenizer.json"),
            weights: model_dir.join("model.safetensors"),
            model_dir,
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

impl Gpt2Runtime {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let paths = Gpt2AssetPaths::new(model_dir.as_ref());
        let config = load_config(&paths.config)?;
        let weights = load_weights(&paths.weights, &config)?;
        let model = Gpt2Model::new(config, weights)?;
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
                let next_decoded = self.tokenizer.decode(&output_ids)?;
                if let Some(delta) = next_decoded.strip_prefix(&decoded) {
                    on_text(delta)?;
                } else {
                    let token_text = self.tokenizer.decode(&[token_id])?;
                    on_text(&token_text)?;
                }
                decoded = next_decoded;
                Ok(())
            })?;

        Ok(decoded)
    }
}

pub fn default_gpt2_small_dir() -> PathBuf {
    PathBuf::from("models/gpt2")
}

pub fn download_gpt2_small_assets(model_dir: impl AsRef<Path>) -> Result<Gpt2AssetPaths> {
    download_huggingface_gpt2_assets("gpt2", "main", model_dir)
}

pub fn download_huggingface_gpt2_assets(
    model_id: &str,
    revision: &str,
    model_dir: impl AsRef<Path>,
) -> Result<Gpt2AssetPaths> {
    let paths = Gpt2AssetPaths::new(model_dir.as_ref());
    fs::create_dir_all(&paths.model_dir).map_err(|err| {
        Gpt2Error::Asset(format!(
            "failed to create model dir {}: {err}",
            paths.model_dir.display()
        ))
    })?;

    download_hf_file(model_id, revision, "config.json", &paths.config)?;
    download_hf_file(model_id, revision, "tokenizer.json", &paths.tokenizer)?;
    download_hf_file(model_id, revision, "model.safetensors", &paths.weights)?;

    Ok(paths)
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
    let bytes = fs::read(path).map_err(|err| {
        Gpt2Error::Asset(format!("failed to read weights {}: {err}", path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|err| {
        Gpt2Error::Asset(format!(
            "failed to parse safetensors {}: {err}",
            path.display()
        ))
    })?;

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

fn tensor_f32(tensors: &SafeTensors<'_>, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let prefixed_name = format!("transformer.{name}");
    let tensor = match tensors.tensor(name) {
        Ok(tensor) => tensor,
        Err(err) => tensors.tensor(&prefixed_name).map_err(|prefixed_err| {
            Gpt2Error::Asset(format!(
                "failed to read tensor {name} ({err}) or {prefixed_name} ({prefixed_err})"
            ))
        })?,
    };
    if tensor.dtype() != Dtype::F32 {
        return Err(Gpt2Error::Asset(format!(
            "tensor {name} has dtype {:?}, expected F32",
            tensor.dtype()
        )));
    }
    if tensor.shape() != expected_shape {
        return Err(Gpt2Error::InvalidWeights(format!(
            "tensor {name} shape {:?} does not match expected {:?}",
            tensor.shape(),
            expected_shape
        )));
    }

    let data = tensor.data();
    if data.len() % 4 != 0 {
        return Err(Gpt2Error::Asset(format!(
            "tensor {name} byte length {} is not divisible by 4",
            data.len()
        )));
    }

    Ok(data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn download_hf_file(model_id: &str, revision: &str, filename: &str, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }

    let tmp = dst.with_extension("download");
    let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/{filename}");
    let client = reqwest::blocking::Client::builder()
        .user_agent("puppygrad/0.1")
        .build()
        .map_err(|err| Gpt2Error::Asset(format!("failed to build HTTP client: {err}")))?;
    let mut response = client
        .get(&url)
        .send()
        .and_then(|response| response.error_for_status())
        .map_err(|err| Gpt2Error::Asset(format!("failed to download {url}: {err}")))?;
    let mut file = File::create(&tmp).map_err(|err| {
        Gpt2Error::Asset(format!(
            "failed to create temporary file {}: {err}",
            tmp.display()
        ))
    })?;
    io::copy(&mut response, &mut file).map_err(|err| {
        Gpt2Error::Asset(format!("failed to write download {}: {err}", tmp.display()))
    })?;
    fs::rename(&tmp, dst).map_err(|err| {
        Gpt2Error::Asset(format!(
            "failed to move {} to {}: {err}",
            tmp.display(),
            dst.display()
        ))
    })?;
    Ok(())
}

impl Gpt2Model {
    pub fn new(config: Gpt2Config, weights: Gpt2Weights) -> Result<Self> {
        config.validate()?;
        validate_weights(&config, &weights)?;
        Ok(Self { config, weights })
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

        for block in &self.weights.blocks {
            x = self.block_forward(&x, t, block);
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
        for pos in 0..t {
            let hidden = row(&x, pos, c);
            let out = row_mut(&mut logits, pos, cfg.vocab_size);
            for (token, logit) in out.iter_mut().enumerate().take(cfg.vocab_size) {
                *logit = dot(hidden, row(&self.weights.wte, token, c));
            }
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
        self.validate_input(input_ids)?;
        cache.check_compatible(&self.config)?;
        cache.clear();

        let mut logits = Vec::new();
        for token_id in input_ids {
            logits = self.forward_one(*token_id, cache)?;
        }
        Ok(logits)
    }

    pub fn forward_one(&self, token_id: usize, cache: &mut Gpt2KvCache) -> Result<Vec<f32>> {
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
        let mut x = vec![0.0f32; c];

        let tok = row(&self.weights.wte, token_id, c);
        let pos_emb = row(&self.weights.wpe, pos, c);
        for i in 0..c {
            x[i] = tok[i] + pos_emb[i];
        }

        for (layer, block) in self.weights.blocks.iter().enumerate() {
            x = self.block_forward_one(&x, layer, block, cache)?;
        }

        layer_norm_in_place(
            &mut x,
            1,
            c,
            &self.weights.ln_f_g,
            &self.weights.ln_f_b,
            cfg.layer_norm_epsilon,
        );
        cache.seq_len += 1;
        Ok(logits_from_hidden(&x, &self.weights.wte, cfg.vocab_size, c))
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
        self.validate_input(input_ids)?;
        let mut cache = self.new_kv_cache()?;
        let mut logits = self.prefill(input_ids, &mut cache)?;
        let mut tokens = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            if tokens.len() >= self.config.n_positions {
                break;
            }
            let next = argmax(&logits);
            tokens.push(next);
            on_token(next)?;
            logits = self.forward_one(next, &mut cache)?;
        }

        Ok(tokens)
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

    fn block_forward(&self, x: &[f32], t: usize, block: &Gpt2BlockWeights) -> Vec<f32> {
        let cfg = &self.config;
        let c = cfg.n_embd;
        let mut norm = x.to_vec();
        layer_norm_in_place(
            &mut norm,
            t,
            c,
            &block.ln_1_g,
            &block.ln_1_b,
            cfg.layer_norm_epsilon,
        );

        let qkv = linear(&norm, t, c, &block.c_attn_w, &block.c_attn_b, 3 * c);
        let attn = causal_self_attention(&qkv, t, cfg.n_head, cfg.head_dim());
        let attn_proj = linear(&attn, t, c, &block.c_proj_w, &block.c_proj_b, c);

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

        let mut mlp = linear(&norm, t, c, &block.c_fc_w, &block.c_fc_b, cfg.n_inner);
        gelu_in_place(&mut mlp);
        let mlp_proj = linear(
            &mlp,
            t,
            cfg.n_inner,
            &block.c_proj_mlp_w,
            &block.c_proj_mlp_b,
            c,
        );

        add_in_place(&mut residual, &mlp_proj);
        residual
    }

    fn block_forward_one(
        &self,
        x: &[f32],
        layer: usize,
        block: &Gpt2BlockWeights,
        cache: &mut Gpt2KvCache,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let c = cfg.n_embd;
        let mut norm = x.to_vec();
        layer_norm_in_place(
            &mut norm,
            1,
            c,
            &block.ln_1_g,
            &block.ln_1_b,
            cfg.layer_norm_epsilon,
        );

        let qkv = linear(&norm, 1, c, &block.c_attn_w, &block.c_attn_b, 3 * c);
        let attn = cached_self_attention(
            &qkv,
            cache.seq_len,
            &mut cache.layers[layer],
            cfg.n_head,
            cfg.head_dim(),
        );
        let attn_proj = linear(&attn, 1, c, &block.c_proj_w, &block.c_proj_b, c);

        let mut residual = x.to_vec();
        add_in_place(&mut residual, &attn_proj);

        let mut norm = residual.clone();
        layer_norm_in_place(
            &mut norm,
            1,
            c,
            &block.ln_2_g,
            &block.ln_2_b,
            cfg.layer_norm_epsilon,
        );

        let mut mlp = linear(&norm, 1, c, &block.c_fc_w, &block.c_fc_b, cfg.n_inner);
        gelu_in_place(&mut mlp);
        let mlp_proj = linear(
            &mlp,
            1,
            cfg.n_inner,
            &block.c_proj_mlp_w,
            &block.c_proj_mlp_b,
            c,
        );

        add_in_place(&mut residual, &mlp_proj);
        Ok(residual)
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

fn linear(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    out_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        let src = row(x, r, in_features);
        let dst = row_mut(&mut out, r, out_features);
        for o in 0..out_features {
            let mut sum = bias[o];
            for i in 0..in_features {
                sum += src[i] * weight[i * out_features + o];
            }
            dst[o] = sum;
        }
    }
    out
}

fn causal_self_attention(qkv: &[f32], seq_len: usize, n_head: usize, head_dim: usize) -> Vec<f32> {
    let n_embd = n_head * head_dim;
    let mut out = vec![0.0f32; seq_len * n_embd];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

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

fn cached_self_attention(
    qkv: &[f32],
    pos: usize,
    cache: &mut Gpt2LayerKvCache,
    n_head: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_embd = n_head * head_dim;
    let mut out = vec![0.0f32; n_embd];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for h in 0..n_head {
        for d in 0..head_dim {
            let idx = kv_index(pos, h, d, n_head, head_dim);
            cache.keys[idx] = qkv[n_embd + h * head_dim + d];
            cache.values[idx] = qkv[2 * n_embd + h * head_dim + d];
        }

        let mut scores = vec![0.0f32; pos + 1];
        for (k_pos, score) in scores.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for d in 0..head_dim {
                let q = qkv[h * head_dim + d];
                let k = cache.keys[kv_index(k_pos, h, d, n_head, head_dim)];
                sum += q * k;
            }
            *score = sum * scale;
        }
        softmax_in_place(&mut scores);

        for (k_pos, prob) in scores.iter().copied().enumerate() {
            for d in 0..head_dim {
                let v = cache.values[kv_index(k_pos, h, d, n_head, head_dim)];
                out[h * head_dim + d] += prob * v;
            }
        }
    }

    out
}

fn logits_from_hidden(
    hidden: &[f32],
    token_embeddings: &[f32],
    vocab_size: usize,
    n_embd: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];
    for (token, logit) in logits.iter_mut().enumerate() {
        *logit = dot(hidden, row(token_embeddings, token, n_embd));
    }
    logits
}

fn kv_index(pos: usize, head: usize, dim: usize, n_head: usize, head_dim: usize) -> usize {
    (pos * n_head + head) * head_dim + dim
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn row(values: &[f32], row: usize, cols: usize) -> &[f32] {
    &values[row * cols..(row + 1) * cols]
}

fn row_mut(values: &mut [f32], row: usize, cols: usize) -> &mut [f32] {
    &mut values[row * cols..(row + 1) * cols]
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
