use std::collections::BTreeMap;

use super::{PiperError, PiperPhonemeType, PiperVoiceConfig, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct PhonemePhrase {
    pub phonemes: Vec<char>,
    pub trailing_silence_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiperPhonemeMap {
    replacements: BTreeMap<char, Vec<char>>,
    ids: BTreeMap<char, Vec<usize>>,
}

impl PiperPhonemeMap {
    pub fn from_config(config: &PiperVoiceConfig) -> Result<Self> {
        config.validate()?;
        let mut replacements = BTreeMap::new();
        for (from, to) in &config.phoneme_map {
            let from = single_char(from)?;
            let to = to
                .iter()
                .map(|value| single_char(value))
                .collect::<Result<Vec<_>>>()?;
            replacements.insert(from, to);
        }

        let mut ids = BTreeMap::new();
        for (phoneme, phoneme_ids) in &config.phoneme_id_map {
            ids.insert(single_char(phoneme)?, phoneme_ids.clone());
        }

        Ok(Self { replacements, ids })
    }

    pub fn ids_for_phoneme(&self, phoneme: char) -> Option<&[usize]> {
        self.ids.get(&phoneme).map(Vec::as_slice)
    }

    pub fn phoneme_ids(&self, phonemes: &[char]) -> Result<Vec<usize>> {
        let mut out = Vec::new();
        let mut missing = Vec::new();
        for phoneme in phonemes {
            if let Some(mapped) = self.replacements.get(phoneme) {
                for replacement in mapped {
                    self.push_ids(*replacement, &mut out, &mut missing);
                }
            } else {
                self.push_ids(*phoneme, &mut out, &mut missing);
            }
        }

        if !missing.is_empty() {
            missing.sort_unstable();
            missing.dedup();
            return Err(PiperError::InvalidInput(format!(
                "missing phoneme id mapping for {}",
                missing.into_iter().collect::<String>()
            )));
        }
        Ok(out)
    }

    fn push_ids(&self, phoneme: char, out: &mut Vec<usize>, missing: &mut Vec<char>) {
        if let Some(ids) = self.ids_for_phoneme(phoneme) {
            out.extend_from_slice(ids);
        } else {
            missing.push(phoneme);
        }
    }
}

pub fn phoneme_ids(config: &PiperVoiceConfig, phonemes: &[char]) -> Result<Vec<usize>> {
    PiperPhonemeMap::from_config(config)?.phoneme_ids(phonemes)
}

pub fn text_to_phonemes(config: &PiperVoiceConfig, text: &str) -> Result<Vec<char>> {
    match config.phoneme_type {
        PiperPhonemeType::Text => Ok(text.chars().collect()),
        PiperPhonemeType::Espeak => Err(PiperError::Unsupported(
            "Piper eSpeak phonemization is not linked yet; pass --phoneme-ids or use a text-mode voice"
                .to_string(),
        )),
    }
}

pub fn split_phoneme_phrases(
    config: &PiperVoiceConfig,
    phonemes: &[char],
) -> Result<Vec<PhonemePhrase>> {
    let mut silence = BTreeMap::new();
    for (key, seconds) in &config.inference.phoneme_silence {
        if *seconds < 0.0 {
            return Err(PiperError::InvalidConfig(format!(
                "phoneme_silence for {key:?} must be >= 0"
            )));
        }
        silence.insert(single_char(key)?, *seconds);
    }

    let mut phrases = Vec::new();
    let mut current = Vec::new();
    for phoneme in phonemes {
        current.push(*phoneme);
        if let Some(seconds) = silence.get(phoneme) {
            phrases.push(PhonemePhrase {
                phonemes: std::mem::take(&mut current),
                trailing_silence_seconds: *seconds,
            });
        }
    }
    if !current.is_empty() || phrases.is_empty() {
        phrases.push(PhonemePhrase {
            phonemes: current,
            trailing_silence_seconds: 0.0,
        });
    }
    Ok(phrases)
}

fn single_char(value: &str) -> Result<char> {
    let mut chars = value.chars();
    let Some(ch) = chars.next() else {
        return Err(PiperError::InvalidConfig(
            "phoneme map entries must not be empty".to_string(),
        ));
    };
    if chars.next().is_some() {
        return Err(PiperError::InvalidConfig(format!(
            "phoneme map entry {value:?} must be one Unicode scalar"
        )));
    }
    Ok(ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::piper::{PiperAudioConfig, PiperInferenceConfig, PiperPhonemeType};

    fn test_config() -> PiperVoiceConfig {
        PiperVoiceConfig {
            audio: PiperAudioConfig::default(),
            inference: PiperInferenceConfig::default(),
            phoneme_type: PiperPhonemeType::Text,
            phoneme_map: BTreeMap::from([('x'.to_string(), vec!["a".to_string()])]),
            phoneme_id_map: BTreeMap::from([
                ("a".to_string(), vec![1]),
                ("b".to_string(), vec![2, 3]),
            ]),
            num_symbols: 4,
            num_speakers: 1,
            speaker_id_map: BTreeMap::new(),
            espeak: None,
            language: None,
            dataset: None,
            piper_version: None,
        }
    }

    #[test]
    fn converts_phonemes_to_ids_with_replacements() {
        let config = test_config();

        let ids = phoneme_ids(&config, &['x', 'b']).unwrap();

        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn reports_missing_phonemes() {
        let config = test_config();

        let err = phoneme_ids(&config, &['z']).unwrap_err();

        assert!(matches!(err, PiperError::InvalidInput(_)));
    }

    #[test]
    fn text_mode_uses_codepoints_as_phonemes() {
        let config = test_config();

        let phonemes = text_to_phonemes(&config, "ab").unwrap();

        assert_eq!(phonemes, vec!['a', 'b']);
    }

    #[test]
    fn splits_phrases_at_configured_silence_phonemes() {
        let mut config = test_config();
        config
            .inference
            .phoneme_silence
            .insert(",".to_string(), 0.2);

        let phrases = split_phoneme_phrases(&config, &['a', ',', 'b']).unwrap();

        assert_eq!(
            phrases,
            vec![
                PhonemePhrase {
                    phonemes: vec!['a', ','],
                    trailing_silence_seconds: 0.2
                },
                PhonemePhrase {
                    phonemes: vec!['b'],
                    trailing_silence_seconds: 0.0
                }
            ]
        );
    }
}
