use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::{Result, StableDiffusionError};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StableDiffusionModelIndex {
    #[serde(rename = "_class_name", default)]
    pub class_name: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ClipTextConfig {
    #[serde(rename = "_class_name", default)]
    pub class_name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub layer_norm_eps: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Unet2DConditionConfig {
    #[serde(rename = "_class_name", default)]
    pub class_name: String,
    pub sample_size: Option<usize>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub layers_per_block: serde_json::Value,
    pub cross_attention_dim: usize,
    #[serde(default)]
    pub attention_head_dim: serde_json::Value,
    #[serde(default)]
    pub transformer_layers_per_block: serde_json::Value,
    #[serde(default)]
    pub norm_num_groups: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AutoencoderKlConfig {
    #[serde(rename = "_class_name", default)]
    pub class_name: String,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    #[serde(default)]
    pub layers_per_block: Option<usize>,
    #[serde(default = "default_vae_scaling_factor")]
    pub scaling_factor: f32,
    #[serde(default)]
    pub norm_num_groups: Option<usize>,
}

fn default_vae_scaling_factor() -> f32 {
    0.18215
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct DdimSchedulerConfig {
    #[serde(rename = "_class_name", default)]
    pub class_name: String,
    pub num_train_timesteps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub beta_schedule: String,
    #[serde(default)]
    pub clip_sample: bool,
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String,
    #[serde(default)]
    pub timestep_spacing: Option<String>,
    #[serde(default)]
    pub steps_offset: Option<usize>,
}

fn default_prediction_type() -> String {
    "epsilon".to_string()
}

pub fn load_model_index(path: &Path) -> Result<StableDiffusionModelIndex> {
    read_json(path)
}

pub fn load_clip_text_config(path: &Path) -> Result<ClipTextConfig> {
    let config: ClipTextConfig = read_json(path)?;
    validate_clip_text_config(&config)?;
    Ok(config)
}

pub fn load_unet_config(path: &Path) -> Result<Unet2DConditionConfig> {
    let config: Unet2DConditionConfig = read_json(path)?;
    validate_unet_config(&config)?;
    Ok(config)
}

pub fn load_vae_config(path: &Path) -> Result<AutoencoderKlConfig> {
    let config: AutoencoderKlConfig = read_json(path)?;
    validate_vae_config(&config)?;
    Ok(config)
}

pub fn load_scheduler_config(path: &Path) -> Result<DdimSchedulerConfig> {
    let config: DdimSchedulerConfig = read_json(path)?;
    validate_scheduler_config(&config)?;
    Ok(config)
}

pub fn validate_clip_text_config(config: &ClipTextConfig) -> Result<()> {
    if !config.class_name.is_empty() && config.class_name != "CLIPTextModel" {
        return Err(StableDiffusionError::Unsupported(format!(
            "text_encoder class {} is not supported; expected CLIPTextModel",
            config.class_name
        )));
    }
    for (name, value) in [
        ("vocab_size", config.vocab_size),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("max_position_embeddings", config.max_position_embeddings),
    ] {
        if value == 0 {
            return Err(StableDiffusionError::Config(format!("{name} must be > 0")));
        }
    }
    if config.hidden_size % config.num_attention_heads != 0 {
        return Err(StableDiffusionError::Config(
            "hidden_size must be divisible by num_attention_heads".to_string(),
        ));
    }
    if !matches!(config.hidden_act.as_str(), "" | "quick_gelu" | "gelu") {
        return Err(StableDiffusionError::Unsupported(format!(
            "CLIP activation {} is not supported",
            config.hidden_act
        )));
    }
    Ok(())
}

pub fn validate_unet_config(config: &Unet2DConditionConfig) -> Result<()> {
    if !config.class_name.is_empty() && config.class_name != "UNet2DConditionModel" {
        return Err(StableDiffusionError::Unsupported(format!(
            "UNet class {} is not supported; expected UNet2DConditionModel",
            config.class_name
        )));
    }
    if config.in_channels != 4 || config.out_channels != 4 {
        return Err(StableDiffusionError::Unsupported(format!(
            "SD 1.x native path expects UNet 4 input and 4 output channels, got {} and {}",
            config.in_channels, config.out_channels
        )));
    }
    if config.block_out_channels.is_empty()
        || config.down_block_types.is_empty()
        || config.up_block_types.is_empty()
    {
        return Err(StableDiffusionError::Config(
            "UNet block channel/type lists must not be empty".to_string(),
        ));
    }
    if config
        .down_block_types
        .iter()
        .any(|block| block.contains("Inpaint"))
    {
        return Err(StableDiffusionError::Unsupported(
            "inpainting UNets are outside the initial SD 1.x text-to-image scope".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_vae_config(config: &AutoencoderKlConfig) -> Result<()> {
    if !config.class_name.is_empty() && config.class_name != "AutoencoderKL" {
        return Err(StableDiffusionError::Unsupported(format!(
            "VAE class {} is not supported; expected AutoencoderKL",
            config.class_name
        )));
    }
    if config.latent_channels != 4 {
        return Err(StableDiffusionError::Unsupported(format!(
            "SD 1.x native path expects 4 VAE latent channels, got {}",
            config.latent_channels
        )));
    }
    if !config.scaling_factor.is_finite() || config.scaling_factor <= 0.0 {
        return Err(StableDiffusionError::Config(
            "VAE scaling_factor must be finite and > 0".to_string(),
        ));
    }
    if matches!(config.layers_per_block, Some(0)) {
        return Err(StableDiffusionError::Config(
            "VAE layers_per_block must be > 0".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_scheduler_config(config: &DdimSchedulerConfig) -> Result<()> {
    if !config.class_name.is_empty()
        && config.class_name != "DDIMScheduler"
        && config.class_name != "PNDMScheduler"
    {
        return Err(StableDiffusionError::Unsupported(format!(
            "scheduler class {} is not supported by the initial native DDIM path",
            config.class_name
        )));
    }
    if config.num_train_timesteps == 0 {
        return Err(StableDiffusionError::Config(
            "num_train_timesteps must be > 0".to_string(),
        ));
    }
    if !matches!(
        config.beta_schedule.as_str(),
        "linear" | "scaled_linear" | "squaredcos_cap_v2"
    ) {
        return Err(StableDiffusionError::Unsupported(format!(
            "scheduler beta_schedule {} is not supported",
            config.beta_schedule
        )));
    }
    if config.prediction_type != "epsilon" && config.prediction_type != "v_prediction" {
        return Err(StableDiffusionError::Unsupported(format!(
            "scheduler prediction_type {} is not supported",
            config.prediction_type
        )));
    }
    if let Some(spacing) = config.timestep_spacing.as_deref() {
        if !matches!(spacing, "leading" | "trailing" | "linspace") {
            return Err(StableDiffusionError::Unsupported(format!(
                "scheduler timestep_spacing {spacing} is not supported"
            )));
        }
    }
    Ok(())
}

fn read_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let bytes = fs::read(path).map_err(|err| {
        StableDiffusionError::Asset(format!("failed to read {}: {err}", path.display()))
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        StableDiffusionError::Config(format!("failed to parse {}: {err}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_clip_shape_fields() {
        let config = ClipTextConfig {
            class_name: "CLIPTextModel".to_string(),
            vocab_size: 10,
            hidden_size: 8,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            max_position_embeddings: 77,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
        };

        assert!(validate_clip_text_config(&config).is_ok());
    }

    #[test]
    fn rejects_unsupported_clip_class() {
        let config = ClipTextConfig {
            class_name: "T5EncoderModel".to_string(),
            vocab_size: 10,
            hidden_size: 8,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            max_position_embeddings: 77,
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-5,
        };

        assert!(validate_clip_text_config(&config)
            .unwrap_err()
            .to_string()
            .contains("T5EncoderModel"));
    }

    #[test]
    fn rejects_unsupported_scheduler() {
        let config = DdimSchedulerConfig {
            class_name: "EulerDiscreteScheduler".to_string(),
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            clip_sample: false,
            prediction_type: "epsilon".to_string(),
            timestep_spacing: None,
            steps_offset: None,
        };

        assert!(validate_scheduler_config(&config)
            .unwrap_err()
            .to_string()
            .contains("EulerDiscreteScheduler"));
    }
}
