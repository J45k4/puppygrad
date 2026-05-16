use serde::{Deserialize, Serialize};

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2GenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_id: Option<usize>,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Gpt2GenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            eos_token_id: Some(50_256),
            temperature: 0.0,
            top_p: None,
            top_k: None,
            seed: 299_792_458,
            repeat_penalty: 1.0,
            repeat_last_n: 128,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Gpt2Error::InvalidConfig(
                "generation temperature must be finite and >= 0".to_string(),
            ));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(Gpt2Error::InvalidConfig(
                    "generation top_p must be finite and in (0, 1]".to_string(),
                ));
            }
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(Gpt2Error::InvalidConfig(
                    "generation top_k must be > 0".to_string(),
                ));
            }
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(Gpt2Error::InvalidConfig(
                "generation repeat_penalty must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}
