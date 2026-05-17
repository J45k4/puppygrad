use std::path::{Path, PathBuf};

use crate::models::assets::{default_model_dir, prepare_huggingface_model_dir, HuggingFaceAsset};

use super::{Result, WhisperError, WhisperSize};

pub const WHISPER_ASSETS: [HuggingFaceAsset; 4] = [
    HuggingFaceAsset::required("config.json"),
    HuggingFaceAsset::required("tokenizer.json"),
    HuggingFaceAsset::required("preprocessor_config.json"),
    HuggingFaceAsset::required("model.safetensors"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperAssetPaths {
    pub model_dir: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub preprocessor_config: PathBuf,
    pub weights: PathBuf,
}

impl WhisperAssetPaths {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        Self {
            config: model_dir.join("config.json"),
            tokenizer: model_dir.join("tokenizer.json"),
            preprocessor_config: model_dir.join("preprocessor_config.json"),
            weights: model_dir.join("model.safetensors"),
            model_dir,
        }
    }
}

pub fn default_whisper_dir(size: WhisperSize) -> PathBuf {
    default_model_dir(size.local_dir_name())
}

pub fn prepare_whisper_assets(
    size: WhisperSize,
    model_id: Option<&str>,
    revision: &str,
    model_dir: impl AsRef<Path>,
    download: bool,
) -> Result<WhisperAssetPaths> {
    let paths = WhisperAssetPaths::new(model_dir.as_ref());
    let model_id = model_id.unwrap_or_else(|| size.model_id());
    prepare_huggingface_model_dir(
        model_id,
        revision,
        &paths.model_dir,
        &WHISPER_ASSETS,
        download,
    )
    .map_err(|err| WhisperError::Asset(err.to_string()))?;
    Ok(paths)
}
