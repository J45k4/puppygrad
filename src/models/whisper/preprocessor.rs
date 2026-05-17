use serde::Deserialize;
use std::path::Path;

use crate::models::config::load_json_config;

use super::{Result, WhisperError};

pub const WHISPER_SAMPLE_RATE: usize = 16_000;
pub const WHISPER_CHUNK_SECONDS: usize = 30;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_N_SAMPLES: usize = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS;
pub const WHISPER_N_FRAMES: usize = WHISPER_N_SAMPLES / WHISPER_HOP_LENGTH;

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperPreprocessorConfig {
    pub sample_rate: usize,
    pub chunk_length_seconds: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub n_samples: usize,
    pub n_frames: usize,
    pub padding_value: f32,
    pub return_attention_mask: bool,
}

impl Default for WhisperPreprocessorConfig {
    fn default() -> Self {
        Self {
            sample_rate: WHISPER_SAMPLE_RATE,
            chunk_length_seconds: WHISPER_CHUNK_SECONDS,
            n_fft: WHISPER_N_FFT,
            hop_length: WHISPER_HOP_LENGTH,
            n_mels: 80,
            n_samples: WHISPER_N_SAMPLES,
            n_frames: WHISPER_N_FRAMES,
            padding_value: 0.0,
            return_attention_mask: false,
        }
    }
}

impl WhisperPreprocessorConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("sample_rate", self.sample_rate),
            ("chunk_length_seconds", self.chunk_length_seconds),
            ("n_fft", self.n_fft),
            ("hop_length", self.hop_length),
            ("n_mels", self.n_mels),
            ("n_samples", self.n_samples),
            ("n_frames", self.n_frames),
        ] {
            if value == 0 {
                return Err(WhisperError::InvalidConfig(format!("{name} must be > 0")));
            }
        }
        if self.sample_rate != WHISPER_SAMPLE_RATE {
            return Err(WhisperError::InvalidConfig(format!(
                "Whisper sample_rate must be {WHISPER_SAMPLE_RATE}, got {}",
                self.sample_rate
            )));
        }
        if self.n_fft != WHISPER_N_FFT {
            return Err(WhisperError::InvalidConfig(format!(
                "Whisper n_fft must be {WHISPER_N_FFT}, got {}",
                self.n_fft
            )));
        }
        if self.hop_length != WHISPER_HOP_LENGTH {
            return Err(WhisperError::InvalidConfig(format!(
                "Whisper hop_length must be {WHISPER_HOP_LENGTH}, got {}",
                self.hop_length
            )));
        }
        let expected_samples = self.sample_rate * self.chunk_length_seconds;
        if self.n_samples != expected_samples {
            return Err(WhisperError::InvalidConfig(format!(
                "n_samples {} must equal sample_rate * chunk_length_seconds {expected_samples}",
                self.n_samples
            )));
        }
        let expected_frames = self.n_samples / self.hop_length;
        if self.n_frames != expected_frames {
            return Err(WhisperError::InvalidConfig(format!(
                "n_frames {} must equal n_samples / hop_length {expected_frames}",
                self.n_frames
            )));
        }
        if self.padding_value != 0.0 {
            return Err(WhisperError::InvalidConfig(format!(
                "padding_value must be 0.0, got {}",
                self.padding_value
            )));
        }
        if self.return_attention_mask {
            return Err(WhisperError::InvalidConfig(
                "return_attention_mask must be false for the native Whisper path".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct HfWhisperPreprocessorConfig {
    #[serde(rename = "sampling_rate")]
    sample_rate: usize,
    #[serde(rename = "chunk_length")]
    chunk_length_seconds: usize,
    #[serde(rename = "n_fft")]
    n_fft: usize,
    hop_length: usize,
    #[serde(rename = "feature_size")]
    n_mels: usize,
    n_samples: Option<usize>,
    #[serde(rename = "nb_max_frames")]
    n_frames: Option<usize>,
    padding_value: Option<f32>,
    return_attention_mask: Option<bool>,
}

pub fn load_whisper_preprocessor_config(
    path: impl AsRef<Path>,
) -> Result<WhisperPreprocessorConfig> {
    let hf: HfWhisperPreprocessorConfig =
        load_json_config(path).map_err(|err| WhisperError::Asset(err.to_string()))?;
    let n_samples = hf
        .n_samples
        .unwrap_or(hf.sample_rate * hf.chunk_length_seconds);
    let n_frames = hf.n_frames.unwrap_or(n_samples / hf.hop_length);
    let config = WhisperPreprocessorConfig {
        sample_rate: hf.sample_rate,
        chunk_length_seconds: hf.chunk_length_seconds,
        n_fft: hf.n_fft,
        hop_length: hf.hop_length,
        n_mels: hf.n_mels,
        n_samples,
        n_frames,
        padding_value: hf.padding_value.unwrap_or(0.0),
        return_attention_mask: hf.return_attention_mask.unwrap_or(false),
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_huggingface_preprocessor_config() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "puppygrad-whisper-preprocessor-{}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{
                "chunk_length": 30,
                "feature_size": 80,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_value": 0.0,
                "return_attention_mask": false,
                "sampling_rate": 16000
            }"#,
        )
        .unwrap();

        let config = load_whisper_preprocessor_config(&path)?;

        assert_eq!(config.sample_rate, WHISPER_SAMPLE_RATE);
        assert_eq!(config.n_samples, WHISPER_N_SAMPLES);
        assert_eq!(config.n_frames, WHISPER_N_FRAMES);
        let _ = fs::remove_file(path);
        Ok(())
    }
}
