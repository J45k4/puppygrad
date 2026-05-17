use std::collections::BTreeMap;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::models::streaming::{RawTokenDecoder, TokenDecoder};

use super::{Result, WhisperError, WhisperSize};

#[derive(Clone)]
pub struct WhisperTokenizer {
    tokenizer: Tokenizer,
    special_tokens: WhisperSpecialTokens,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperSpecialTokens {
    pub start_of_transcript: usize,
    pub eos: usize,
    pub transcribe: Option<usize>,
    pub translate: Option<usize>,
    pub no_timestamps: Option<usize>,
    pub no_speech: Option<usize>,
    pub language_tokens: BTreeMap<String, usize>,
    pub timestamp_begin: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

impl WhisperTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref()).map_err(|err| {
            WhisperError::Asset(format!(
                "failed to load Whisper tokenizer {}: {err}",
                path.as_ref().display()
            ))
        })?;
        let special_tokens = WhisperSpecialTokens::from_tokenizer(&tokenizer)?;
        Ok(Self {
            tokenizer,
            special_tokens,
        })
    }

    pub fn special_tokens(&self) -> &WhisperSpecialTokens {
        &self.special_tokens
    }

    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let token_ids: std::result::Result<Vec<u32>, _> =
            token_ids.iter().map(|id| u32::try_from(*id)).collect();
        let token_ids = token_ids
            .map_err(|_| WhisperError::InvalidInput("token id does not fit in u32".to_string()))?;
        self.tokenizer
            .decode(&token_ids, true)
            .map_err(|err| WhisperError::Asset(format!("failed to decode Whisper tokens: {err}")))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| WhisperError::Asset(format!("failed to encode Whisper text: {err}")))?;
        Ok(encoding.get_ids().iter().map(|id| *id as usize).collect())
    }

    pub fn prompt_prefix(
        &self,
        task: WhisperTask,
        language: Option<&str>,
        no_timestamps: bool,
        size: WhisperSize,
    ) -> Result<Vec<usize>> {
        let mut tokens = vec![self.special_tokens.start_of_transcript];
        if let Some(language) = language {
            if size.is_english_only() && language != "en" {
                return Err(WhisperError::InvalidInput(format!(
                    "language {language:?} is not supported by English-only checkpoint {size}"
                )));
            }
            if !size.is_english_only() {
                let token = self
                    .special_tokens
                    .language_tokens
                    .get(language)
                    .copied()
                    .ok_or_else(|| {
                        WhisperError::InvalidInput(format!(
                            "language {language:?} is not present in tokenizer"
                        ))
                    })?;
                tokens.push(token);
            }
        }
        match task {
            WhisperTask::Transcribe => {
                if let Some(token) = self.special_tokens.transcribe {
                    tokens.push(token);
                }
            }
            WhisperTask::Translate => {
                let token = self.special_tokens.translate.ok_or_else(|| {
                    WhisperError::InvalidInput(
                        "translate task is not available in this tokenizer".to_string(),
                    )
                })?;
                tokens.push(token);
            }
        }
        if no_timestamps {
            let token = self.special_tokens.no_timestamps.ok_or_else(|| {
                WhisperError::InvalidInput(
                    "no-timestamps token is not available in this tokenizer".to_string(),
                )
            })?;
            tokens.push(token);
        }
        Ok(tokens)
    }
}

impl TokenDecoder for WhisperTokenizer {
    type Error = WhisperError;

    fn decode_tokens(&self, token_ids: &[usize]) -> Result<String> {
        self.decode(token_ids)
    }
}

impl RawTokenDecoder for WhisperTokenizer {
    type Error = WhisperError;

    fn raw_token(&self, token_id: usize) -> Result<String> {
        let token_id = u32::try_from(token_id)
            .map_err(|_| WhisperError::InvalidInput("token id does not fit in u32".to_string()))?;
        self.tokenizer
            .id_to_token(token_id)
            .ok_or_else(|| WhisperError::InvalidInput(format!("unknown token id {token_id}")))
    }
}

impl WhisperSpecialTokens {
    fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        let start_of_transcript = token_id(tokenizer, "<|startoftranscript|>")?;
        let eos = token_id(tokenizer, "<|endoftext|>")?;
        let transcribe = maybe_token_id(tokenizer, "<|transcribe|>");
        let translate = maybe_token_id(tokenizer, "<|translate|>");
        let no_timestamps = maybe_token_id(tokenizer, "<|notimestamps|>");
        let no_speech = maybe_token_id(tokenizer, "<|nospeech|>");
        let timestamp_begin = maybe_token_id(tokenizer, "<|0.00|>");
        let mut language_tokens = BTreeMap::new();
        for token in tokenizer.get_vocab(true).keys() {
            if token.starts_with("<|")
                && token.ends_with("|>")
                && token.len() == 6
                && token.as_bytes()[2].is_ascii_lowercase()
                && token.as_bytes()[3].is_ascii_lowercase()
            {
                let language = token[2..4].to_string();
                if let Some(id) = maybe_token_id(tokenizer, token) {
                    language_tokens.insert(language, id);
                }
            }
        }
        Ok(Self {
            start_of_transcript,
            eos,
            transcribe,
            translate,
            no_timestamps,
            no_speech,
            language_tokens,
            timestamp_begin,
        })
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<usize> {
    maybe_token_id(tokenizer, token).ok_or_else(|| {
        WhisperError::Asset(format!(
            "Whisper tokenizer is missing required special token {token}"
        ))
    })
}

fn maybe_token_id(tokenizer: &Tokenizer, token: &str) -> Option<usize> {
    tokenizer.token_to_id(token).map(|id| id as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn builds_prompt_prefixes_from_special_tokens() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "puppygrad-whisper-tokenizer-vocab-{}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{
                "<unk>": 0,
                "<|endoftext|>": 1,
                "<|startoftranscript|>": 2,
                "<|en|>": 3,
                "<|transcribe|>": 4,
                "<|translate|>": 5,
                "<|notimestamps|>": 6,
                "<|nospeech|>": 7,
                "<|0.00|>": 8,
                "hello": 9
            }"#,
        )
        .unwrap();
        let model = WordLevel::from_file(path.to_str().unwrap(), "<unk>".to_string())
            .map_err(|err| WhisperError::Asset(err.to_string()))?;
        let tokenizer = Tokenizer::new(model);
        let special_tokens = WhisperSpecialTokens::from_tokenizer(&tokenizer)?;
        let tokenizer = WhisperTokenizer {
            tokenizer,
            special_tokens,
        };

        assert_eq!(
            tokenizer.prompt_prefix(
                WhisperTask::Transcribe,
                Some("en"),
                true,
                WhisperSize::Tiny
            )?,
            vec![2, 3, 4, 6]
        );
        assert_eq!(
            tokenizer.prompt_prefix(
                WhisperTask::Translate,
                Some("en"),
                false,
                WhisperSize::Tiny
            )?,
            vec![2, 3, 5]
        );
        assert_eq!(
            tokenizer.prompt_prefix(
                WhisperTask::Transcribe,
                Some("en"),
                true,
                WhisperSize::TinyEn
            )?,
            vec![2, 4, 6]
        );
        assert_eq!(tokenizer.decode(&[9])?, "hello");
        assert_eq!(tokenizer.encode("hello")?, vec![9]);

        let _ = fs::remove_file(path);
        Ok(())
    }
}
