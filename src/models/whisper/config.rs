use clap::ValueEnum;
use serde::Deserialize;
use std::fmt;
use std::path::Path;

use crate::models::config::load_json_config;

use super::{Result, WhisperError};

pub const WHISPER_AUDIO_CTX: usize = 1_500;
pub const WHISPER_TEXT_CTX: usize = 448;
pub const WHISPER_MULTILINGUAL_VOCAB: usize = 51_865;
pub const WHISPER_ENGLISH_VOCAB: usize = 51_864;
pub const WHISPER_LARGE_V3_VOCAB: usize = 51_866;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WhisperSize {
    #[value(name = "tiny.en", alias = "tiny-en")]
    TinyEn,
    Tiny,
    #[value(name = "base.en", alias = "base-en")]
    BaseEn,
    Base,
    #[value(name = "small.en", alias = "small-en")]
    SmallEn,
    Small,
    #[value(name = "medium.en", alias = "medium-en")]
    MediumEn,
    Medium,
    #[value(name = "large-v1")]
    LargeV1,
    #[value(name = "large-v2")]
    LargeV2,
    #[value(name = "large-v3", alias = "large")]
    LargeV3,
    #[value(name = "turbo", alias = "large-v3-turbo")]
    Turbo,
}

impl WhisperSize {
    pub fn model_id(self) -> &'static str {
        match self {
            WhisperSize::TinyEn => "openai/whisper-tiny.en",
            WhisperSize::Tiny => "openai/whisper-tiny",
            WhisperSize::BaseEn => "openai/whisper-base.en",
            WhisperSize::Base => "openai/whisper-base",
            WhisperSize::SmallEn => "openai/whisper-small.en",
            WhisperSize::Small => "openai/whisper-small",
            WhisperSize::MediumEn => "openai/whisper-medium.en",
            WhisperSize::Medium => "openai/whisper-medium",
            WhisperSize::LargeV1 => "openai/whisper-large",
            WhisperSize::LargeV2 => "openai/whisper-large-v2",
            WhisperSize::LargeV3 => "openai/whisper-large-v3",
            WhisperSize::Turbo => "openai/whisper-large-v3-turbo",
        }
    }

    pub fn local_dir_name(self) -> &'static str {
        match self {
            WhisperSize::TinyEn => "whisper-tiny.en",
            WhisperSize::Tiny => "whisper-tiny",
            WhisperSize::BaseEn => "whisper-base.en",
            WhisperSize::Base => "whisper-base",
            WhisperSize::SmallEn => "whisper-small.en",
            WhisperSize::Small => "whisper-small",
            WhisperSize::MediumEn => "whisper-medium.en",
            WhisperSize::Medium => "whisper-medium",
            WhisperSize::LargeV1 => "whisper-large-v1",
            WhisperSize::LargeV2 => "whisper-large-v2",
            WhisperSize::LargeV3 => "whisper-large-v3",
            WhisperSize::Turbo => "whisper-large-v3-turbo",
        }
    }

    pub fn is_english_only(self) -> bool {
        matches!(
            self,
            WhisperSize::TinyEn
                | WhisperSize::BaseEn
                | WhisperSize::SmallEn
                | WhisperSize::MediumEn
        )
    }

    pub fn approx_parameters_millions(self) -> usize {
        match self {
            WhisperSize::Tiny | WhisperSize::TinyEn => 39,
            WhisperSize::Base | WhisperSize::BaseEn => 74,
            WhisperSize::Small | WhisperSize::SmallEn => 244,
            WhisperSize::Medium | WhisperSize::MediumEn => 769,
            WhisperSize::LargeV1 | WhisperSize::LargeV2 | WhisperSize::LargeV3 => 1_550,
            WhisperSize::Turbo => 809,
        }
    }

    pub fn approx_vram_gb(self) -> usize {
        match self {
            WhisperSize::Tiny | WhisperSize::TinyEn => 1,
            WhisperSize::Base | WhisperSize::BaseEn => 1,
            WhisperSize::Small | WhisperSize::SmallEn => 2,
            WhisperSize::Medium | WhisperSize::MediumEn => 5,
            WhisperSize::LargeV1 | WhisperSize::LargeV2 | WhisperSize::LargeV3 => 10,
            WhisperSize::Turbo => 6,
        }
    }

    pub fn relative_speed(self) -> &'static str {
        match self {
            WhisperSize::Tiny | WhisperSize::TinyEn => "~10x",
            WhisperSize::Base | WhisperSize::BaseEn => "~7x",
            WhisperSize::Small | WhisperSize::SmallEn => "~4x",
            WhisperSize::Medium | WhisperSize::MediumEn => "~2x",
            WhisperSize::LargeV1 | WhisperSize::LargeV2 | WhisperSize::LargeV3 => "1x",
            WhisperSize::Turbo => "~8x",
        }
    }

    pub fn config(self) -> WhisperConfig {
        let english_only = self.is_english_only();
        let vocab = if english_only {
            WHISPER_ENGLISH_VOCAB
        } else {
            WHISPER_MULTILINGUAL_VOCAB
        };
        match self {
            WhisperSize::Tiny | WhisperSize::TinyEn => WhisperConfig::new(80, 384, 6, 4, 4, vocab),
            WhisperSize::Base | WhisperSize::BaseEn => WhisperConfig::new(80, 512, 8, 6, 6, vocab),
            WhisperSize::Small | WhisperSize::SmallEn => {
                WhisperConfig::new(80, 768, 12, 12, 12, vocab)
            }
            WhisperSize::Medium | WhisperSize::MediumEn => {
                WhisperConfig::new(80, 1_024, 16, 24, 24, vocab)
            }
            WhisperSize::LargeV1 | WhisperSize::LargeV2 => {
                WhisperConfig::new(80, 1_280, 20, 32, 32, WHISPER_MULTILINGUAL_VOCAB)
            }
            WhisperSize::LargeV3 => {
                WhisperConfig::new(128, 1_280, 20, 32, 32, WHISPER_LARGE_V3_VOCAB)
            }
            WhisperSize::Turbo => WhisperConfig::new(128, 1_280, 20, 32, 4, WHISPER_LARGE_V3_VOCAB),
        }
    }
}

impl fmt::Display for WhisperSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.local_dir_name().trim_start_matches("whisper-"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_audio_mlp: usize,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    pub n_text_mlp: usize,
}

impl WhisperConfig {
    pub fn new(
        n_mels: usize,
        n_state: usize,
        n_head: usize,
        n_audio_layer: usize,
        n_text_layer: usize,
        n_vocab: usize,
    ) -> Self {
        Self {
            n_mels,
            n_audio_ctx: WHISPER_AUDIO_CTX,
            n_audio_state: n_state,
            n_audio_head: n_head,
            n_audio_layer,
            n_audio_mlp: 4 * n_state,
            n_vocab,
            n_text_ctx: WHISPER_TEXT_CTX,
            n_text_state: n_state,
            n_text_head: n_head,
            n_text_layer,
            n_text_mlp: 4 * n_state,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("n_mels", self.n_mels),
            ("n_audio_ctx", self.n_audio_ctx),
            ("n_audio_state", self.n_audio_state),
            ("n_audio_head", self.n_audio_head),
            ("n_audio_layer", self.n_audio_layer),
            ("n_audio_mlp", self.n_audio_mlp),
            ("n_vocab", self.n_vocab),
            ("n_text_ctx", self.n_text_ctx),
            ("n_text_state", self.n_text_state),
            ("n_text_head", self.n_text_head),
            ("n_text_layer", self.n_text_layer),
            ("n_text_mlp", self.n_text_mlp),
        ] {
            if value == 0 {
                return Err(WhisperError::InvalidConfig(format!("{name} must be > 0")));
            }
        }
        if !self.n_audio_state.is_multiple_of(self.n_audio_head) {
            return Err(WhisperError::InvalidConfig(format!(
                "n_audio_state {} must be divisible by n_audio_head {}",
                self.n_audio_state, self.n_audio_head
            )));
        }
        if !self.n_text_state.is_multiple_of(self.n_text_head) {
            return Err(WhisperError::InvalidConfig(format!(
                "n_text_state {} must be divisible by n_text_head {}",
                self.n_text_state, self.n_text_head
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct HfWhisperConfig {
    vocab_size: usize,
    num_mel_bins: usize,
    max_source_positions: usize,
    max_target_positions: usize,
    d_model: usize,
    encoder_attention_heads: usize,
    encoder_layers: usize,
    encoder_ffn_dim: usize,
    decoder_attention_heads: usize,
    decoder_layers: usize,
    decoder_ffn_dim: usize,
}

pub fn load_whisper_config(path: impl AsRef<Path>) -> Result<WhisperConfig> {
    let hf: HfWhisperConfig =
        load_json_config(path).map_err(|err| WhisperError::Asset(err.to_string()))?;
    let config = WhisperConfig {
        n_mels: hf.num_mel_bins,
        n_audio_ctx: hf.max_source_positions,
        n_audio_state: hf.d_model,
        n_audio_head: hf.encoder_attention_heads,
        n_audio_layer: hf.encoder_layers,
        n_audio_mlp: hf.encoder_ffn_dim,
        n_vocab: hf.vocab_size,
        n_text_ctx: hf.max_target_positions,
        n_text_state: hf.d_model,
        n_text_head: hf.decoder_attention_heads,
        n_text_layer: hf.decoder_layers,
        n_text_mlp: hf.decoder_ffn_dim,
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn tiny_preset_matches_expected_shape() -> Result<()> {
        let config = WhisperSize::Tiny.config();

        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_audio_layer, 4);
        assert_eq!(config.n_text_layer, 4);
        assert_eq!(config.n_audio_state, 384);
        assert_eq!(config.n_audio_head, 6);
        assert_eq!(config.n_vocab, WHISPER_MULTILINGUAL_VOCAB);
        config.validate()
    }

    #[test]
    fn turbo_preset_keeps_large_encoder_and_small_decoder() -> Result<()> {
        let config = WhisperSize::Turbo.config();

        assert_eq!(config.n_mels, 128);
        assert_eq!(config.n_audio_layer, 32);
        assert_eq!(config.n_text_layer, 4);
        assert_eq!(config.n_audio_state, 1_280);
        assert_eq!(config.n_audio_head, 20);
        assert_eq!(config.n_vocab, WHISPER_LARGE_V3_VOCAB);
        config.validate()
    }

    #[test]
    fn english_only_presets_use_english_vocab() {
        assert_eq!(WhisperSize::TinyEn.config().n_vocab, WHISPER_ENGLISH_VOCAB);
        assert_eq!(WhisperSize::BaseEn.config().n_vocab, WHISPER_ENGLISH_VOCAB);
    }

    #[test]
    fn parses_representative_huggingface_configs() -> Result<()> {
        for (name, json, expected) in [
            (
                "tiny",
                hf_config_json(51_865, 80, 384, 6, 4, 4, 1_536),
                WhisperSize::Tiny.config(),
            ),
            (
                "base",
                hf_config_json(51_865, 80, 512, 8, 6, 6, 2_048),
                WhisperSize::Base.config(),
            ),
            (
                "small",
                hf_config_json(51_865, 80, 768, 12, 12, 12, 3_072),
                WhisperSize::Small.config(),
            ),
            (
                "medium",
                hf_config_json(51_865, 80, 1_024, 16, 24, 24, 4_096),
                WhisperSize::Medium.config(),
            ),
            (
                "large-v3",
                hf_config_json(51_866, 128, 1_280, 20, 32, 32, 5_120),
                WhisperSize::LargeV3.config(),
            ),
            (
                "turbo",
                hf_config_json(51_866, 128, 1_280, 20, 32, 4, 5_120),
                WhisperSize::Turbo.config(),
            ),
        ] {
            let path = std::env::temp_dir().join(format!(
                "puppygrad-whisper-{name}-config-{}.json",
                std::process::id()
            ));
            fs::write(&path, json).unwrap();

            let actual = load_whisper_config(&path)?;

            assert_eq!(actual, expected);
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    #[test]
    fn size_presets_define_expected_encoder_output_shapes() -> Result<()> {
        for size in [
            WhisperSize::TinyEn,
            WhisperSize::Tiny,
            WhisperSize::BaseEn,
            WhisperSize::Base,
            WhisperSize::SmallEn,
            WhisperSize::Small,
            WhisperSize::MediumEn,
            WhisperSize::Medium,
            WhisperSize::LargeV1,
            WhisperSize::LargeV2,
            WhisperSize::LargeV3,
            WhisperSize::Turbo,
        ] {
            let config = size.config();
            config.validate()?;
            assert_eq!(config.n_audio_ctx, WHISPER_AUDIO_CTX);
            assert_eq!(config.n_audio_state % config.n_audio_head, 0);
        }
        Ok(())
    }

    fn hf_config_json(
        vocab_size: usize,
        num_mel_bins: usize,
        d_model: usize,
        heads: usize,
        encoder_layers: usize,
        decoder_layers: usize,
        ffn_dim: usize,
    ) -> String {
        format!(
            r#"{{
                "vocab_size": {vocab_size},
                "num_mel_bins": {num_mel_bins},
                "max_source_positions": 1500,
                "max_target_positions": 448,
                "d_model": {d_model},
                "encoder_attention_heads": {heads},
                "encoder_layers": {encoder_layers},
                "encoder_ffn_dim": {ffn_dim},
                "decoder_attention_heads": {heads},
                "decoder_layers": {decoder_layers},
                "decoder_ffn_dim": {ffn_dim}
            }}"#
        )
    }
}
