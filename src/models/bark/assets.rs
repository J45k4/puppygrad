use std::path::{Path, PathBuf};

use crate::models::assets::{default_model_dir, prepare_huggingface_model_dir, HuggingFaceAsset};

use super::{BarkError, Result};

pub const BARK_CONFIG_JSON: &str = "config.json";
pub const BARK_GENERATION_CONFIG_JSON: &str = "generation_config.json";
pub const BARK_TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
pub const BARK_SPECIAL_TOKENS_MAP_JSON: &str = "special_tokens_map.json";
pub const BARK_VOCAB_TXT: &str = "vocab.txt";
pub const BARK_PYTORCH_WEIGHTS: &str = "pytorch_model.bin";

pub const BARK_SMALL_MODEL_ID: &str = "suno/bark-small";

pub const BARK_ASSETS: [HuggingFaceAsset; 6] = [
    HuggingFaceAsset::required(BARK_CONFIG_JSON),
    HuggingFaceAsset::required(BARK_GENERATION_CONFIG_JSON),
    HuggingFaceAsset::required(BARK_TOKENIZER_CONFIG_JSON),
    HuggingFaceAsset::required(BARK_SPECIAL_TOKENS_MAP_JSON),
    HuggingFaceAsset::required(BARK_VOCAB_TXT),
    HuggingFaceAsset::required(BARK_PYTORCH_WEIGHTS),
];

pub const BARK_METADATA_ASSETS: [HuggingFaceAsset; 5] = [
    HuggingFaceAsset::required(BARK_CONFIG_JSON),
    HuggingFaceAsset::required(BARK_GENERATION_CONFIG_JSON),
    HuggingFaceAsset::required(BARK_TOKENIZER_CONFIG_JSON),
    HuggingFaceAsset::required(BARK_SPECIAL_TOKENS_MAP_JSON),
    HuggingFaceAsset::required(BARK_VOCAB_TXT),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkAssetPaths {
    pub model_dir: PathBuf,
    pub config: PathBuf,
    pub generation_config: PathBuf,
    pub tokenizer_config: PathBuf,
    pub special_tokens_map: PathBuf,
    pub vocab: PathBuf,
    pub weights: PathBuf,
}

impl BarkAssetPaths {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        Self {
            config: model_dir.join(BARK_CONFIG_JSON),
            generation_config: model_dir.join(BARK_GENERATION_CONFIG_JSON),
            tokenizer_config: model_dir.join(BARK_TOKENIZER_CONFIG_JSON),
            special_tokens_map: model_dir.join(BARK_SPECIAL_TOKENS_MAP_JSON),
            vocab: model_dir.join(BARK_VOCAB_TXT),
            weights: model_dir.join(BARK_PYTORCH_WEIGHTS),
            model_dir,
        }
    }
}

pub fn default_bark_dir() -> PathBuf {
    default_model_dir("bark-small")
}

pub fn prepare_bark_assets(
    model_id: &str,
    revision: &str,
    model_dir: impl AsRef<Path>,
    download: bool,
) -> Result<BarkAssetPaths> {
    let paths = BarkAssetPaths::new(model_dir.as_ref());
    prepare_huggingface_model_dir(model_id, revision, &paths.model_dir, &BARK_ASSETS, download)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    Ok(paths)
}

pub fn prepare_bark_metadata_assets(
    model_id: &str,
    revision: &str,
    model_dir: impl AsRef<Path>,
    download: bool,
) -> Result<BarkAssetPaths> {
    let paths = BarkAssetPaths::new(model_dir.as_ref());
    prepare_huggingface_model_dir(
        model_id,
        revision,
        &paths.model_dir,
        &BARK_METADATA_ASSETS,
        download,
    )
    .map_err(|err| BarkError::Asset(err.to_string()))?;
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_default_asset_paths() {
        let paths = BarkAssetPaths::new("models/bark-small");

        assert_eq!(paths.model_dir, PathBuf::from("models/bark-small"));
        assert_eq!(paths.config, PathBuf::from("models/bark-small/config.json"));
        assert_eq!(paths.vocab, PathBuf::from("models/bark-small/vocab.txt"));
        assert_eq!(
            paths.weights,
            PathBuf::from("models/bark-small/pytorch_model.bin")
        );
    }
}
