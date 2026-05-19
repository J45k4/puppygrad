use super::config::{default_resnet18_dir, ResNetConfig};
use super::model::ResNetClassification;
use super::rust::ResNet18Rust;
use super::weights::{load_resnet18_weights, ResNetWeightError};
use crate::models::assets::{
    check_required_files, download_huggingface_file, download_huggingface_files, AssetError,
    HuggingFaceAsset,
};
use crate::vision::cnn::{softmax, top_k};
use crate::vision::{load_rgb8, preprocess_rgb8_to_normalized_chw, ChwImage, VisionError};
use image::imageops::FilterType;
use serde_json::Value;
use std::error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub const RESNET18_HF_MODEL_ID: &str = "timm/resnet18.tv_in1k";
pub const RESNET18_REVISION: &str = "main";
const LABEL_MODEL_ID: &str = "datasets/huggingface/label-files";
const LABEL_FILENAME: &str = "imagenet-1k-id2label.json";
const RESNET18_ASSETS: [HuggingFaceAsset; 2] = [
    HuggingFaceAsset::required("config.json"),
    HuggingFaceAsset::required("model.safetensors"),
];

#[derive(Debug)]
pub enum ResNetRuntimeError {
    Asset(AssetError),
    Vision(VisionError),
    Weights(ResNetWeightError),
    ReadLabels {
        path: String,
        source: std::io::Error,
    },
    ParseLabels {
        path: String,
        source: serde_json::Error,
    },
    InvalidLabels {
        path: String,
        message: String,
    },
}

impl fmt::Display for ResNetRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResNetRuntimeError::Asset(source) => write!(f, "{source}"),
            ResNetRuntimeError::Vision(source) => write!(f, "{source}"),
            ResNetRuntimeError::Weights(source) => write!(f, "{source}"),
            ResNetRuntimeError::ReadLabels { path, source } => {
                write!(f, "failed to read labels {path}: {source}")
            }
            ResNetRuntimeError::ParseLabels { path, source } => {
                write!(f, "failed to parse labels {path}: {source}")
            }
            ResNetRuntimeError::InvalidLabels { path, message } => {
                write!(f, "invalid labels {path}: {message}")
            }
        }
    }
}

impl error::Error for ResNetRuntimeError {}

impl From<AssetError> for ResNetRuntimeError {
    fn from(value: AssetError) -> Self {
        Self::Asset(value)
    }
}

impl From<VisionError> for ResNetRuntimeError {
    fn from(value: VisionError) -> Self {
        Self::Vision(value)
    }
}

impl From<ResNetWeightError> for ResNetRuntimeError {
    fn from(value: ResNetWeightError) -> Self {
        Self::Weights(value)
    }
}

pub type Result<T> = std::result::Result<T, ResNetRuntimeError>;

#[derive(Clone, Debug)]
pub struct ResNetRuntime {
    pub config: ResNetConfig,
    model: ResNet18Rust,
    labels: Vec<String>,
}

impl ResNetRuntime {
    pub fn from_dir(model_dir: &Path, labels_path: Option<&Path>) -> Result<Self> {
        check_resnet18_assets(model_dir)?;
        let config = ResNetConfig::resnet18_imagenet();
        let weights = load_resnet18_weights(&model_dir.join("model.safetensors"), &config)?;
        let labels = load_labels(labels_path.unwrap_or(&default_labels_path(model_dir)))?;
        Ok(Self {
            config,
            model: ResNet18Rust::new(weights),
            labels,
        })
    }

    pub fn logits_for_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        let image = preprocess_resnet_image(image_path, &self.config)?;
        Ok(self.model.logits(&image.data, image.height, image.width))
    }

    pub fn classify_image(&self, image_path: &Path, k: usize) -> Result<Vec<ResNetClassification>> {
        let logits = self.logits_for_image(image_path)?;
        Ok(classifications_from_logits(&logits, &self.labels, k))
    }
}

pub fn classifications_from_logits(
    logits: &[f32],
    labels: &[String],
    k: usize,
) -> Vec<ResNetClassification> {
    let probabilities = softmax(logits);
    top_k(&probabilities, k)
        .into_iter()
        .map(|(class_index, probability)| ResNetClassification {
            class_index,
            label: labels
                .get(class_index)
                .cloned()
                .unwrap_or_else(|| format!("class {class_index}")),
            probability,
            logit: logits[class_index],
        })
        .collect()
}

pub fn preprocess_resnet_image(image_path: &Path, config: &ResNetConfig) -> Result<ChwImage> {
    let image = load_rgb8(image_path)?;
    Ok(preprocess_rgb8_to_normalized_chw(
        &image,
        config.resize_short_side,
        config.crop_size,
        config.mean,
        config.std,
        FilterType::Triangle,
    )?)
}

pub fn download_resnet18_assets(model_dir: &Path) -> Result<()> {
    download_huggingface_files(
        RESNET18_HF_MODEL_ID,
        RESNET18_REVISION,
        model_dir,
        &RESNET18_ASSETS,
    )?;
    download_huggingface_file(
        LABEL_MODEL_ID,
        "main",
        LABEL_FILENAME,
        &model_dir.join(LABEL_FILENAME),
    )?;
    Ok(())
}

pub fn check_resnet18_assets(model_dir: &Path) -> Result<()> {
    check_required_files(model_dir, &["model.safetensors"])?;
    let labels = default_labels_path(model_dir);
    if !labels.is_file() {
        check_required_files(model_dir, &[LABEL_FILENAME])?;
    }
    Ok(())
}

pub fn default_model_dir() -> PathBuf {
    default_resnet18_dir()
}

pub fn default_labels_path(model_dir: &Path) -> PathBuf {
    let labels_json = model_dir.join("labels.json");
    if labels_json.is_file() {
        return labels_json;
    }
    let labels_txt = model_dir.join("labels.txt");
    if labels_txt.is_file() {
        return labels_txt;
    }
    model_dir.join(LABEL_FILENAME)
}

pub fn load_labels(path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).map_err(|source| ResNetRuntimeError::ReadLabels {
        path: path.display().to_string(),
        source,
    })?;
    if path.extension().and_then(|ext| ext.to_str()) == Some("txt") {
        let labels = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        validate_labels(path, labels)
    } else {
        let value: Value =
            serde_json::from_str(&text).map_err(|source| ResNetRuntimeError::ParseLabels {
                path: path.display().to_string(),
                source,
            })?;
        match value {
            Value::Array(values) => validate_labels(
                path,
                values
                    .into_iter()
                    .map(|value| value.as_str().unwrap_or_default().to_string())
                    .collect(),
            ),
            Value::Object(map) => {
                let mut labels = Vec::with_capacity(map.len());
                for index in 0..map.len() {
                    let Some(value) = map.get(&index.to_string()) else {
                        return Err(ResNetRuntimeError::InvalidLabels {
                            path: path.display().to_string(),
                            message: format!("missing label index {index}"),
                        });
                    };
                    labels.push(value.as_str().unwrap_or_default().to_string());
                }
                validate_labels(path, labels)
            }
            _ => Err(ResNetRuntimeError::InvalidLabels {
                path: path.display().to_string(),
                message: "expected JSON object, JSON array, or text file".to_string(),
            }),
        }
    }
}

fn validate_labels(path: &Path, labels: Vec<String>) -> Result<Vec<String>> {
    if labels.len() != 1000 || labels.iter().any(|label| label.is_empty()) {
        return Err(ResNetRuntimeError::InvalidLabels {
            path: path.display().to_string(),
            message: format!("expected 1000 non-empty labels, got {}", labels.len()),
        });
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifications_include_probabilities_labels_and_logits() {
        let labels = (0..1000)
            .map(|index| format!("label-{index}"))
            .collect::<Vec<_>>();
        let mut logits = vec![0.0; 1000];
        logits[3] = 2.0;
        logits[7] = 3.0;

        let top = classifications_from_logits(&logits, &labels, 2);

        assert_eq!(top[0].class_index, 7);
        assert_eq!(top[0].label, "label-7");
        assert_eq!(top[0].logit, 3.0);
        assert!(top[0].probability > top[1].probability);
        assert_eq!(top[1].class_index, 3);
    }
}
