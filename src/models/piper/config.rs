use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::models::config::load_json_config;

use super::{PiperError, Result};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PiperPhonemeType {
    Espeak,
    Text,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PiperAudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_sample_bytes")]
    pub sample_bytes: usize,
    #[serde(default = "default_channels")]
    pub channels: usize,
    pub quality: Option<String>,
}

impl Default for PiperAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            sample_bytes: default_sample_bytes(),
            channels: default_channels(),
            quality: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PiperInferenceConfig {
    #[serde(default = "default_noise_scale")]
    pub noise_scale: f32,
    #[serde(default = "default_length_scale")]
    pub length_scale: f32,
    #[serde(default = "default_noise_w")]
    pub noise_w: f32,
    #[serde(default)]
    pub phoneme_silence: BTreeMap<String, f32>,
}

impl Default for PiperInferenceConfig {
    fn default() -> Self {
        Self {
            noise_scale: default_noise_scale(),
            length_scale: default_length_scale(),
            noise_w: default_noise_w(),
            phoneme_silence: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct PiperEspeakConfig {
    pub voice: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct PiperLanguageConfig {
    pub code: Option<String>,
    pub family: Option<String>,
    pub region: Option<String>,
    pub name_native: Option<String>,
    pub name_english: Option<String>,
    pub country_english: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PiperVoiceConfig {
    #[serde(default)]
    pub audio: PiperAudioConfig,
    #[serde(default)]
    pub inference: PiperInferenceConfig,
    pub phoneme_type: PiperPhonemeType,
    #[serde(default)]
    pub phoneme_map: BTreeMap<String, Vec<String>>,
    pub phoneme_id_map: BTreeMap<String, Vec<usize>>,
    pub num_symbols: usize,
    pub num_speakers: usize,
    #[serde(default)]
    pub speaker_id_map: BTreeMap<String, usize>,
    pub espeak: Option<PiperEspeakConfig>,
    pub language: Option<PiperLanguageConfig>,
    pub dataset: Option<String>,
    pub piper_version: Option<String>,
}

impl PiperVoiceConfig {
    pub fn validate(&self) -> Result<()> {
        if self.audio.sample_rate == 0 {
            return Err(PiperError::InvalidConfig(
                "audio.sample_rate must be > 0".to_string(),
            ));
        }
        if self.audio.channels == 0 {
            return Err(PiperError::InvalidConfig(
                "audio.channels must be > 0".to_string(),
            ));
        }
        if self.num_symbols == 0 {
            return Err(PiperError::InvalidConfig(
                "num_symbols must be > 0".to_string(),
            ));
        }
        if self.num_speakers == 0 {
            return Err(PiperError::InvalidConfig(
                "num_speakers must be > 0".to_string(),
            ));
        }
        if self.phoneme_id_map.is_empty() {
            return Err(PiperError::InvalidConfig(
                "phoneme_id_map must not be empty".to_string(),
            ));
        }
        for key in self.phoneme_id_map.keys().chain(self.phoneme_map.keys()) {
            if key.chars().count() != 1 {
                return Err(PiperError::InvalidConfig(format!(
                    "phoneme map key {key:?} must be a single Unicode scalar"
                )));
            }
        }
        for (phoneme, ids) in &self.phoneme_id_map {
            if ids.is_empty() {
                return Err(PiperError::InvalidConfig(format!(
                    "phoneme {phoneme:?} maps to no ids"
                )));
            }
            for id in ids {
                if *id >= self.num_symbols {
                    return Err(PiperError::InvalidConfig(format!(
                        "phoneme {phoneme:?} maps to id {id}, but num_symbols is {}",
                        self.num_symbols
                    )));
                }
            }
        }
        if self.num_speakers > 1 && self.speaker_id_map.is_empty() {
            return Err(PiperError::InvalidConfig(
                "multi-speaker voices must provide speaker_id_map".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn load_piper_config(path: impl AsRef<Path>) -> Result<PiperVoiceConfig> {
    let config: PiperVoiceConfig =
        load_json_config(path).map_err(|err| PiperError::InvalidConfig(err.to_string()))?;
    config.validate()?;
    Ok(config)
}

fn default_sample_rate() -> usize {
    22_050
}

fn default_sample_bytes() -> usize {
    2
}

fn default_channels() -> usize {
    1
}

fn default_noise_scale() -> f32 {
    0.667
}

fn default_length_scale() -> f32 {
    1.0
}

fn default_noise_w() -> f32 {
    0.8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_piper_voice_config() {
        let config: PiperVoiceConfig = serde_json::from_str(
            r#"{
                "audio": {"sample_rate": 22050, "quality": "medium"},
                "espeak": {"voice": "en-us"},
                "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
                "phoneme_type": "espeak",
                "phoneme_map": {},
                "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]},
                "num_symbols": 4,
                "num_speakers": 1,
                "speaker_id_map": {},
                "piper_version": "1.0.0"
            }"#,
        )
        .unwrap();

        config.validate().unwrap();
        assert_eq!(config.phoneme_type, PiperPhonemeType::Espeak);
        assert_eq!(config.audio.sample_rate, 22_050);
        assert_eq!(config.inference.noise_w, 0.8);
        assert_eq!(config.espeak.unwrap().voice.as_deref(), Some("en-us"));
    }

    #[test]
    fn rejects_out_of_range_phoneme_ids() {
        let config: PiperVoiceConfig = serde_json::from_str(
            r#"{
                "phoneme_type": "text",
                "phoneme_id_map": {"a": [3]},
                "num_symbols": 3,
                "num_speakers": 1
            }"#,
        )
        .unwrap();

        let err = config.validate().unwrap_err();
        assert!(matches!(err, PiperError::InvalidConfig(_)));
    }
}
