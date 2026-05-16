use std::path::{Path, PathBuf};

use super::{Gpt2Error, Result};
use crate::models::assets::download_huggingface_file;

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
    download_huggingface_file(model_id, revision, "config.json", &paths.config)
        .map_err(|err| Gpt2Error::Asset(err.to_string()))?;
    download_huggingface_file(model_id, revision, "tokenizer.json", &paths.tokenizer)
        .map_err(|err| Gpt2Error::Asset(err.to_string()))?;
    download_huggingface_file(model_id, revision, "model.safetensors", &paths.weights)
        .map_err(|err| Gpt2Error::Asset(err.to_string()))?;

    Ok(paths)
}
