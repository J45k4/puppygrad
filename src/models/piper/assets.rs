use std::path::{Path, PathBuf};

use crate::models::assets::check_required_files;

use super::{load_piper_config, PiperError, PiperVoiceConfig, Result};

pub const PIPER_CONFIG_JSON: &str = "config.json";
pub const PIPER_ONNX_CONFIG_JSON: &str = "model.onnx.json";
pub const PIPER_ONNX_MODEL: &str = "model.onnx";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiperAssetPaths {
    pub model_dir: PathBuf,
    pub config: PathBuf,
    pub onnx: PathBuf,
}

impl PiperAssetPaths {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        let config = default_config_path(&model_dir);
        let onnx = model_dir.join(PIPER_ONNX_MODEL);
        Self {
            model_dir,
            config,
            onnx,
        }
    }
}

pub fn default_piper_dir() -> PathBuf {
    PathBuf::from("models/piper")
}

pub fn check_piper_assets(model_dir: &Path) -> Result<PiperAssetPaths> {
    let paths = PiperAssetPaths::new(model_dir);
    if !paths.config.is_file() {
        return Err(PiperError::InvalidConfig(format!(
            "Piper voice config not found; expected {} or {} in {}",
            PIPER_CONFIG_JSON,
            PIPER_ONNX_CONFIG_JSON,
            model_dir.display()
        )));
    }
    check_required_files(model_dir, &[PIPER_ONNX_MODEL])
        .map_err(|err| PiperError::InvalidConfig(err.to_string()))?;
    Ok(paths)
}

pub fn load_piper_voice_config_from_dir(model_dir: &Path) -> Result<PiperVoiceConfig> {
    let paths = PiperAssetPaths::new(model_dir);
    if !paths.config.is_file() {
        return Err(PiperError::InvalidConfig(format!(
            "Piper voice config not found; expected {} or {} in {}",
            PIPER_CONFIG_JSON,
            PIPER_ONNX_CONFIG_JSON,
            model_dir.display()
        )));
    }
    load_piper_config(paths.config)
}

fn default_config_path(model_dir: &Path) -> PathBuf {
    let config = model_dir.join(PIPER_CONFIG_JSON);
    if config.is_file() {
        return config;
    }
    model_dir.join(PIPER_ONNX_CONFIG_JSON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn builds_default_asset_paths() {
        let paths = PiperAssetPaths::new("models/piper-test");

        assert_eq!(paths.model_dir, PathBuf::from("models/piper-test"));
        assert_eq!(
            paths.config,
            PathBuf::from("models/piper-test").join(PIPER_ONNX_CONFIG_JSON)
        );
        assert_eq!(
            paths.onnx,
            PathBuf::from("models/piper-test").join(PIPER_ONNX_MODEL)
        );
    }

    #[test]
    fn loads_config_from_voice_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "puppygrad-piper-assets-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(
            tmp.join(PIPER_CONFIG_JSON),
            r#"{
                "phoneme_type": "text",
                "phoneme_id_map": {"a": [1]},
                "num_symbols": 2,
                "num_speakers": 1
            }"#,
        )
        .unwrap();

        let config = load_piper_voice_config_from_dir(&tmp).unwrap();

        assert_eq!(config.num_symbols, 2);
        let _ = fs::remove_dir_all(&tmp);
    }
}
