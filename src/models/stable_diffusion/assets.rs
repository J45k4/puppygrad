use std::path::{Path, PathBuf};

use crate::models::assets::{default_model_dir, prepare_huggingface_model_dir, HuggingFaceAsset};

use super::{Result, StableDiffusionError};

pub const STABLE_DIFFUSION_V1_5_MODEL_ID: &str = "runwayml/stable-diffusion-v1-5";
pub const STABLE_DIFFUSION_TINY_TEST_MODEL_ID: &str =
    "hf-internal-testing/tiny-stable-diffusion-pipe";

pub const MODEL_INDEX_JSON: &str = "model_index.json";
pub const SCHEDULER_CONFIG_JSON: &str = "scheduler/scheduler_config.json";
pub const TOKENIZER_JSON: &str = "tokenizer/tokenizer.json";
pub const TOKENIZER_VOCAB_JSON: &str = "tokenizer/vocab.json";
pub const TOKENIZER_MERGES_TXT: &str = "tokenizer/merges.txt";
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer/tokenizer_config.json";
pub const TEXT_ENCODER_CONFIG_JSON: &str = "text_encoder/config.json";
pub const TEXT_ENCODER_SAFETENSORS: &str = "text_encoder/model.safetensors";
pub const TEXT_ENCODER_PYTORCH_BIN: &str = "text_encoder/pytorch_model.bin";
pub const UNET_CONFIG_JSON: &str = "unet/config.json";
pub const UNET_SAFETENSORS: &str = "unet/diffusion_pytorch_model.safetensors";
pub const UNET_SAFETENSORS_INDEX_JSON: &str = "unet/diffusion_pytorch_model.safetensors.index.json";
pub const UNET_PYTORCH_BIN: &str = "unet/diffusion_pytorch_model.bin";
pub const VAE_CONFIG_JSON: &str = "vae/config.json";
pub const VAE_SAFETENSORS: &str = "vae/diffusion_pytorch_model.safetensors";
pub const VAE_SAFETENSORS_INDEX_JSON: &str = "vae/diffusion_pytorch_model.safetensors.index.json";
pub const VAE_PYTORCH_BIN: &str = "vae/diffusion_pytorch_model.bin";

pub const STABLE_DIFFUSION_NATIVE_REQUIRED_FILES: [&str; 11] = [
    MODEL_INDEX_JSON,
    SCHEDULER_CONFIG_JSON,
    TOKENIZER_VOCAB_JSON,
    TOKENIZER_MERGES_TXT,
    TOKENIZER_CONFIG_JSON,
    TEXT_ENCODER_CONFIG_JSON,
    TEXT_ENCODER_SAFETENSORS,
    UNET_CONFIG_JSON,
    UNET_SAFETENSORS,
    VAE_CONFIG_JSON,
    VAE_SAFETENSORS,
];

pub const STABLE_DIFFUSION_DIFFUSERS_ASSETS: [HuggingFaceAsset; 11] = [
    HuggingFaceAsset::required(MODEL_INDEX_JSON),
    HuggingFaceAsset::required(SCHEDULER_CONFIG_JSON),
    HuggingFaceAsset::required(TOKENIZER_VOCAB_JSON),
    HuggingFaceAsset::required(TOKENIZER_MERGES_TXT),
    HuggingFaceAsset::required(TOKENIZER_CONFIG_JSON),
    HuggingFaceAsset::required(TEXT_ENCODER_CONFIG_JSON),
    HuggingFaceAsset::required(TEXT_ENCODER_SAFETENSORS),
    HuggingFaceAsset::required(UNET_CONFIG_JSON),
    HuggingFaceAsset::required(UNET_SAFETENSORS),
    HuggingFaceAsset::required(VAE_CONFIG_JSON),
    HuggingFaceAsset::required(VAE_SAFETENSORS),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableDiffusionAssetPaths {
    pub model_dir: PathBuf,
    pub model_index: PathBuf,
    pub scheduler_config: PathBuf,
    pub tokenizer_json: PathBuf,
    pub tokenizer_vocab: PathBuf,
    pub tokenizer_merges: PathBuf,
    pub tokenizer_config: PathBuf,
    pub text_encoder_config: PathBuf,
    pub text_encoder_safetensors: PathBuf,
    pub unet_config: PathBuf,
    pub unet_safetensors: PathBuf,
    pub vae_config: PathBuf,
    pub vae_safetensors: PathBuf,
}

impl StableDiffusionAssetPaths {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        Self {
            model_index: model_dir.join(MODEL_INDEX_JSON),
            scheduler_config: model_dir.join(SCHEDULER_CONFIG_JSON),
            tokenizer_json: model_dir.join(TOKENIZER_JSON),
            tokenizer_vocab: model_dir.join(TOKENIZER_VOCAB_JSON),
            tokenizer_merges: model_dir.join(TOKENIZER_MERGES_TXT),
            tokenizer_config: model_dir.join(TOKENIZER_CONFIG_JSON),
            text_encoder_config: model_dir.join(TEXT_ENCODER_CONFIG_JSON),
            text_encoder_safetensors: model_dir.join(TEXT_ENCODER_SAFETENSORS),
            unet_config: model_dir.join(UNET_CONFIG_JSON),
            unet_safetensors: model_dir.join(UNET_SAFETENSORS),
            vae_config: model_dir.join(VAE_CONFIG_JSON),
            vae_safetensors: model_dir.join(VAE_SAFETENSORS),
            model_dir,
        }
    }
}

pub fn default_stable_diffusion_dir() -> PathBuf {
    default_model_dir("stable-diffusion-v1-5")
}

pub fn prepare_stable_diffusion_assets(
    model_id: &str,
    revision: &str,
    model_dir: impl AsRef<Path>,
    download: bool,
) -> Result<StableDiffusionAssetPaths> {
    let paths = StableDiffusionAssetPaths::new(model_dir.as_ref());
    prepare_huggingface_model_dir(
        model_id,
        revision,
        &paths.model_dir,
        &STABLE_DIFFUSION_DIFFUSERS_ASSETS,
        download,
    )
    .map_err(|err| StableDiffusionError::Asset(err.to_string()))?;
    Ok(paths)
}

pub fn validate_stable_diffusion_native_assets(paths: &StableDiffusionAssetPaths) -> Result<()> {
    let missing = STABLE_DIFFUSION_NATIVE_REQUIRED_FILES
        .iter()
        .copied()
        .filter(|filename| !paths.model_dir.join(filename).is_file())
        .collect::<Vec<_>>();
    if missing.is_empty() {
        validate_unsharded_safetensors(paths)?;
        return Ok(());
    }

    let bin_hints = [
        TEXT_ENCODER_PYTORCH_BIN,
        UNET_PYTORCH_BIN,
        VAE_PYTORCH_BIN,
        UNET_SAFETENSORS_INDEX_JSON,
        VAE_SAFETENSORS_INDEX_JSON,
    ]
    .iter()
    .filter(|filename| paths.model_dir.join(filename).is_file())
    .copied()
    .collect::<Vec<_>>();
    let hint = if bin_hints.is_empty() {
        String::new()
    } else {
        format!(
            "; found unsupported native checkpoint files: {}",
            bin_hints.join(", ")
        )
    };

    Err(StableDiffusionError::Asset(format!(
        "native Rust backend requires unsharded safetensors in {}; missing: {}{}",
        paths.model_dir.display(),
        missing.join(", "),
        hint
    )))
}

fn validate_unsharded_safetensors(paths: &StableDiffusionAssetPaths) -> Result<()> {
    for filename in [
        UNET_SAFETENSORS_INDEX_JSON,
        VAE_SAFETENSORS_INDEX_JSON,
        TEXT_ENCODER_PYTORCH_BIN,
        UNET_PYTORCH_BIN,
        VAE_PYTORCH_BIN,
    ] {
        if paths.model_dir.join(filename).is_file() {
            return Err(StableDiffusionError::Unsupported(format!(
                "native Rust backend currently supports only unsharded safetensors; unsupported file present: {}",
                paths.model_dir.join(filename).display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_nested_asset_paths() {
        let paths = StableDiffusionAssetPaths::new("models/sd");

        assert_eq!(
            paths.model_index,
            PathBuf::from("models/sd/model_index.json")
        );
        assert_eq!(
            paths.scheduler_config,
            PathBuf::from("models/sd/scheduler/scheduler_config.json")
        );
        assert_eq!(
            paths.unet_safetensors,
            PathBuf::from("models/sd/unet/diffusion_pytorch_model.safetensors")
        );
    }

    #[test]
    fn missing_native_assets_name_expected_files() {
        let paths = StableDiffusionAssetPaths::new("models/definitely-missing-sd");

        let err = validate_stable_diffusion_native_assets(&paths).unwrap_err();

        assert!(err.to_string().contains("model_index.json"));
        assert!(err.to_string().contains("text_encoder/model.safetensors"));
        assert!(err
            .to_string()
            .contains("unet/diffusion_pytorch_model.safetensors"));
    }
}
