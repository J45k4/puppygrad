use std::path::Path;

use serde::Deserialize;

use crate::models::config::load_json_config;

use super::{BarkError, Result};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkConfig {
    pub semantic_config: BarkSubModelConfig,
    pub coarse_acoustics_config: BarkSubModelConfig,
    pub fine_acoustics_config: BarkFineSubModelConfig,
    pub codec_config: BarkCodecConfig,
    #[serde(default)]
    pub initializer_range: f32,
    pub model_type: Option<String>,
}

impl BarkConfig {
    pub fn validate(&self) -> Result<()> {
        self.semantic_config.validate("semantic_config")?;
        self.coarse_acoustics_config
            .validate("coarse_acoustics_config")?;
        self.fine_acoustics_config
            .validate("fine_acoustics_config")?;
        self.codec_config.validate()?;
        if let Some(model_type) = &self.model_type {
            if model_type != "bark" {
                return Err(BarkError::InvalidConfig(format!(
                    "model_type must be bark, got {model_type:?}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkSubModelConfig {
    pub block_size: usize,
    pub input_vocab_size: usize,
    pub output_vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default)]
    pub bias: bool,
    #[serde(default)]
    pub use_cache: bool,
    pub model_type: Option<String>,
}

impl BarkSubModelConfig {
    pub fn validate(&self, name: &str) -> Result<()> {
        if self.block_size == 0 {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.block_size must be > 0"
            )));
        }
        if self.input_vocab_size == 0 || self.output_vocab_size == 0 {
            return Err(BarkError::InvalidConfig(format!(
                "{name} vocab sizes must be > 0"
            )));
        }
        if self.num_layers == 0 {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.num_layers must be > 0"
            )));
        }
        if self.num_heads == 0 {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.num_heads must be > 0"
            )));
        }
        if self.hidden_size == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.hidden_size must be > 0 and divisible by num_heads"
            )));
        }
        if !self.dropout.is_finite() || self.dropout < 0.0 {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.dropout must be finite and >= 0"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkFineSubModelConfig {
    #[serde(flatten)]
    pub base: BarkSubModelConfig,
    pub n_codes_total: usize,
    pub n_codes_given: usize,
}

impl BarkFineSubModelConfig {
    pub fn validate(&self, name: &str) -> Result<()> {
        self.base.validate(name)?;
        if self.n_codes_total == 0 {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.n_codes_total must be > 0"
            )));
        }
        if self.n_codes_given == 0 || self.n_codes_given > self.n_codes_total {
            return Err(BarkError::InvalidConfig(format!(
                "{name}.n_codes_given must be in 1..=n_codes_total"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkCodecConfig {
    pub sampling_rate: usize,
    pub audio_channels: usize,
    #[serde(default = "default_codec_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_codec_num_filters")]
    pub num_filters: usize,
    #[serde(default = "default_codec_num_residual_layers")]
    pub num_residual_layers: usize,
    pub codebook_size: usize,
    #[serde(default = "default_codec_num_quantizers")]
    pub num_quantizers: usize,
    pub codebook_dim: usize,
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "default_codec_kernel_size")]
    pub kernel_size: usize,
    #[serde(default = "default_codec_last_kernel_size")]
    pub last_kernel_size: usize,
    #[serde(default = "default_codec_residual_kernel_size")]
    pub residual_kernel_size: usize,
    #[serde(default = "default_codec_dilation_growth_rate")]
    pub dilation_growth_rate: usize,
    #[serde(default = "default_codec_compress")]
    pub compress: usize,
    #[serde(default = "default_codec_num_lstm_layers")]
    pub num_lstm_layers: usize,
    #[serde(default = "default_codec_use_causal_conv")]
    pub use_causal_conv: bool,
    #[serde(default = "default_codec_trim_right_ratio")]
    pub trim_right_ratio: f32,
    #[serde(default = "default_codec_norm_type")]
    pub norm_type: String,
    #[serde(default = "default_codec_pad_mode")]
    pub pad_mode: String,
    #[serde(default = "default_codec_use_conv_shortcut")]
    pub use_conv_shortcut: bool,
    pub model_type: Option<String>,
}

impl BarkCodecConfig {
    pub fn validate(&self) -> Result<()> {
        if self.sampling_rate == 0 {
            return Err(BarkError::InvalidConfig(
                "codec_config.sampling_rate must be > 0".to_string(),
            ));
        }
        if self.audio_channels == 0 {
            return Err(BarkError::InvalidConfig(
                "codec_config.audio_channels must be > 0".to_string(),
            ));
        }
        if self.hidden_size == 0
            || self.num_filters == 0
            || self.num_residual_layers == 0
            || self.codebook_size == 0
            || self.num_quantizers == 0
            || self.codebook_dim == 0
            || self.kernel_size == 0
            || self.last_kernel_size == 0
            || self.residual_kernel_size == 0
            || self.dilation_growth_rate == 0
            || self.compress == 0
            || self.num_lstm_layers == 0
        {
            return Err(BarkError::InvalidConfig(
                "codec_config channel, codebook, kernel, and layer sizes must be > 0".to_string(),
            ));
        }
        if self.upsampling_ratios.is_empty() || self.upsampling_ratios.contains(&0) {
            return Err(BarkError::InvalidConfig(
                "codec_config.upsampling_ratios must contain non-zero values".to_string(),
            ));
        }
        if self.norm_type != "weight_norm" {
            return Err(BarkError::InvalidConfig(format!(
                "codec_config.norm_type {:?} is not supported yet; expected weight_norm",
                self.norm_type
            )));
        }
        if self.pad_mode != "reflect" && self.pad_mode != "constant" {
            return Err(BarkError::InvalidConfig(format!(
                "codec_config.pad_mode {:?} is not supported yet",
                self.pad_mode
            )));
        }
        if !self.trim_right_ratio.is_finite()
            || self.trim_right_ratio < 0.0
            || self.trim_right_ratio > 1.0
        {
            return Err(BarkError::InvalidConfig(
                "codec_config.trim_right_ratio must be finite and in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkGenerationConfig {
    pub sample_rate: usize,
    pub codebook_size: usize,
    pub semantic_config: BarkSemanticGenerationConfig,
    pub coarse_acoustics_config: BarkCoarseGenerationConfig,
    pub fine_acoustics_config: BarkFineGenerationConfig,
    pub model_type: Option<String>,
}

impl BarkGenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(BarkError::InvalidConfig(
                "generation_config.sample_rate must be > 0".to_string(),
            ));
        }
        if self.codebook_size == 0 {
            return Err(BarkError::InvalidConfig(
                "generation_config.codebook_size must be > 0".to_string(),
            ));
        }
        self.semantic_config.validate()?;
        self.coarse_acoustics_config.validate()?;
        self.fine_acoustics_config.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkSemanticGenerationConfig {
    pub eos_token_id: usize,
    pub max_input_semantic_length: usize,
    pub max_new_tokens: usize,
    pub semantic_infer_token: usize,
    pub semantic_pad_token: usize,
    pub semantic_rate_hz: f32,
    pub semantic_vocab_size: usize,
    pub text_encoding_offset: usize,
    pub text_pad_token: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

impl BarkSemanticGenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_input_semantic_length == 0 || self.max_new_tokens == 0 {
            return Err(BarkError::InvalidConfig(
                "semantic generation lengths must be > 0".to_string(),
            ));
        }
        if self.semantic_vocab_size == 0 {
            return Err(BarkError::InvalidConfig(
                "semantic_vocab_size must be > 0".to_string(),
            ));
        }
        validate_sampling(self.temperature, self.top_k, self.top_p, "semantic_config")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkCoarseGenerationConfig {
    pub coarse_infer_token: usize,
    pub coarse_rate_hz: usize,
    pub coarse_semantic_pad_token: usize,
    pub max_coarse_history: usize,
    pub max_coarse_input_length: usize,
    pub n_coarse_codebooks: usize,
    pub sliding_window_len: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

impl BarkCoarseGenerationConfig {
    pub fn semantic_to_coarse_token_ratio(&self, semantic_rate_hz: f32) -> f32 {
        self.coarse_rate_hz as f32 / semantic_rate_hz * self.n_coarse_codebooks as f32
    }

    pub fn validate(&self) -> Result<()> {
        if self.coarse_rate_hz == 0
            || self.max_coarse_history == 0
            || self.max_coarse_input_length == 0
            || self.n_coarse_codebooks == 0
            || self.sliding_window_len == 0
        {
            return Err(BarkError::InvalidConfig(
                "coarse generation rates, lengths, and codebook counts must be > 0".to_string(),
            ));
        }
        validate_sampling(
            self.temperature,
            self.top_k,
            self.top_p,
            "coarse_acoustics_config",
        )
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BarkFineGenerationConfig {
    pub max_fine_history_length: usize,
    pub max_fine_input_length: usize,
    pub n_fine_codebooks: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

impl BarkFineGenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_fine_history_length == 0
            || self.max_fine_input_length == 0
            || self.n_fine_codebooks == 0
        {
            return Err(BarkError::InvalidConfig(
                "fine generation lengths and codebook counts must be > 0".to_string(),
            ));
        }
        validate_sampling(
            self.temperature,
            self.top_k,
            self.top_p,
            "fine_acoustics_config",
        )
    }
}

pub fn load_bark_config(path: impl AsRef<Path>) -> Result<BarkConfig> {
    let config: BarkConfig =
        load_json_config(path).map_err(|err| BarkError::InvalidConfig(err.to_string()))?;
    config.validate()?;
    Ok(config)
}

pub fn load_bark_generation_config(path: impl AsRef<Path>) -> Result<BarkGenerationConfig> {
    let config: BarkGenerationConfig =
        load_json_config(path).map_err(|err| BarkError::InvalidConfig(err.to_string()))?;
    config.validate()?;
    Ok(config)
}

fn validate_sampling(temperature: f32, top_k: usize, top_p: f32, name: &str) -> Result<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(BarkError::InvalidConfig(format!(
            "{name}.temperature must be finite and >= 0"
        )));
    }
    if top_k == 0 {
        return Err(BarkError::InvalidConfig(format!(
            "{name}.top_k must be > 0"
        )));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(BarkError::InvalidConfig(format!(
            "{name}.top_p must be finite and in (0, 1]"
        )));
    }
    Ok(())
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> usize {
    50
}

fn default_top_p() -> f32 {
    1.0
}

fn default_codec_hidden_size() -> usize {
    128
}

fn default_codec_num_filters() -> usize {
    32
}

fn default_codec_num_residual_layers() -> usize {
    1
}

fn default_codec_num_quantizers() -> usize {
    8
}

fn default_codec_kernel_size() -> usize {
    7
}

fn default_codec_last_kernel_size() -> usize {
    7
}

fn default_codec_residual_kernel_size() -> usize {
    3
}

fn default_codec_dilation_growth_rate() -> usize {
    2
}

fn default_codec_compress() -> usize {
    2
}

fn default_codec_num_lstm_layers() -> usize {
    2
}

fn default_codec_use_causal_conv() -> bool {
    true
}

fn default_codec_trim_right_ratio() -> f32 {
    1.0
}

fn default_codec_norm_type() -> String {
    "weight_norm".to_string()
}

fn default_codec_pad_mode() -> String {
    "reflect".to_string()
}

fn default_codec_use_conv_shortcut() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_generation_config_shape() {
        let config: BarkGenerationConfig = serde_json::from_str(
            r#"{
              "sample_rate": 24000,
              "codebook_size": 1024,
              "model_type": "bark",
              "semantic_config": {
                "eos_token_id": 10000,
                "max_input_semantic_length": 256,
                "max_new_tokens": 768,
                "semantic_infer_token": 129599,
                "semantic_pad_token": 10000,
                "semantic_rate_hz": 49.9,
                "semantic_vocab_size": 10000,
                "text_encoding_offset": 10048,
                "text_pad_token": 129595,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 1.0
              },
              "coarse_acoustics_config": {
                "coarse_infer_token": 12050,
                "coarse_rate_hz": 75,
                "coarse_semantic_pad_token": 12048,
                "max_coarse_history": 630,
                "max_coarse_input_length": 256,
                "n_coarse_codebooks": 2,
                "sliding_window_len": 60,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 1.0
              },
              "fine_acoustics_config": {
                "max_fine_history_length": 512,
                "max_fine_input_length": 1024,
                "n_fine_codebooks": 8,
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 1.0
              }
            }"#,
        )
        .unwrap();

        config.validate().unwrap();
        assert_eq!(config.sample_rate, 24_000);
        assert_eq!(config.semantic_config.text_encoding_offset, 10_048);
    }

    #[test]
    fn rejects_bad_transformer_shape() {
        let config = BarkSubModelConfig {
            block_size: 1024,
            input_vocab_size: 10,
            output_vocab_size: 10,
            num_layers: 1,
            num_heads: 7,
            hidden_size: 12,
            dropout: 0.0,
            bias: false,
            use_cache: true,
            model_type: Some("semantic".to_string()),
        };

        assert!(config.validate("semantic_config").is_err());
    }
}
