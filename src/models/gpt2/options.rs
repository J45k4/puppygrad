use super::{Gpt2Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gpt2BackendName {
    Rust,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gpt2BackendConfig {
    Rust(Gpt2RustConfig),
}

impl Gpt2BackendConfig {
    pub fn rust(threads: usize) -> Result<Self> {
        let config = Gpt2RustConfig {
            threads,
            ..Gpt2RustConfig::default()
        };
        config.validate()?;
        Ok(Self::Rust(config))
    }

    pub fn name(&self) -> Gpt2BackendName {
        match self {
            Gpt2BackendConfig::Rust(_) => Gpt2BackendName::Rust,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Gpt2BackendConfig::Rust(config) => format!(
                "rust (threads={}, dense_threshold={}, weights={})",
                config.threads,
                config.dense_parallel_threshold,
                if config.quantized_weights {
                    "int8"
                } else {
                    "f32"
                }
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2RustConfig {
    pub threads: usize,
    pub dense_parallel_threshold: usize,
    pub qkv_chunk_size: usize,
    pub attention_projection_chunk_size: usize,
    pub mlp_fc_chunk_size: usize,
    pub mlp_projection_chunk_size: usize,
    pub logits_chunk_size: usize,
    pub attention_head_parallel_threshold: usize,
    pub quantized_weights: bool,
}

impl Default for Gpt2RustConfig {
    fn default() -> Self {
        Self {
            threads: 1,
            dense_parallel_threshold: 262_144,
            qkv_chunk_size: 48,
            attention_projection_chunk_size: 64,
            mlp_fc_chunk_size: 128,
            mlp_projection_chunk_size: 64,
            logits_chunk_size: 256,
            attention_head_parallel_threshold: 4_096,
            quantized_weights: false,
        }
    }
}

impl Gpt2RustConfig {
    pub fn validate(&self) -> Result<()> {
        if self.threads == 0 {
            return Err(Gpt2Error::InvalidConfig(
                "rust backend threads must be > 0".to_string(),
            ));
        }
        if self.dense_parallel_threshold == 0 {
            return Err(Gpt2Error::InvalidConfig(
                "rust dense_parallel_threshold must be > 0".to_string(),
            ));
        }
        for (name, value) in [
            ("qkv_chunk_size", self.qkv_chunk_size),
            (
                "attention_projection_chunk_size",
                self.attention_projection_chunk_size,
            ),
            ("mlp_fc_chunk_size", self.mlp_fc_chunk_size),
            ("mlp_projection_chunk_size", self.mlp_projection_chunk_size),
            ("logits_chunk_size", self.logits_chunk_size),
            (
                "attention_head_parallel_threshold",
                self.attention_head_parallel_threshold,
            ),
        ] {
            if value == 0 {
                return Err(Gpt2Error::InvalidConfig(format!("rust {name} must be > 0")));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2GenerationConfig {
    pub max_new_tokens: usize,
}

impl Gpt2GenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self { max_new_tokens }
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}
