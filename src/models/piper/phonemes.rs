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
        PiperPhonemeType::Espeak => fallback_english_text_to_phonemes(config, text),
    }
}

fn fallback_english_text_to_phonemes(config: &PiperVoiceConfig, text: &str) -> Result<Vec<char>> {
    let map = PiperPhonemeMap::from_config(config)?;
    let mut phonemes = Vec::new();
    push_if_mapped(&map, '^', &mut phonemes);

    let mut word = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            word.push(ch.to_ascii_lowercase());
        } else {
            flush_english_word(&map, &mut word, &mut phonemes)?;
            if ch.is_whitespace() || matches!(ch, '-' | ',' | ';' | ':' | '.' | '!' | '?') {
                push_separator(&map, &mut phonemes);
            }
        }
    }
    flush_english_word(&map, &mut word, &mut phonemes)?;

    while phonemes.last() == Some(&'_') {
        phonemes.pop();
    }
    push_if_mapped(&map, '$', &mut phonemes);
    if phonemes.is_empty() {
        return Err(PiperError::InvalidInput(
            "text input produced no phonemes".to_string(),
        ));
    }
    Ok(phonemes)
}

fn flush_english_word(
    map: &PiperPhonemeMap,
    word: &mut String,
    phonemes: &mut Vec<char>,
) -> Result<()> {
    if word.is_empty() {
        return Ok(());
    }
    if !phonemes.is_empty() && phonemes.last() != Some(&'^') && phonemes.last() != Some(&'_') {
        push_separator(map, phonemes);
    }

    if let Some(pronunciation) = english_dictionary_pronunciation(word) {
        push_pronunciation(map, word, pronunciation, phonemes)?;
    } else {
        let pronunciation = approximate_english_word(word);
        push_pronunciation(map, word, &pronunciation, phonemes)?;
    }
    word.clear();
    Ok(())
}

fn english_dictionary_pronunciation(word: &str) -> Option<&'static str> {
    match word.trim_matches('\'') {
        "a" => Some("ə"),
        "an" => Some("æn"),
        "and" => Some("ænd"),
        "are" => Some("ɑɹ"),
        "as" => Some("æz"),
        "be" => Some("bi"),
        "for" => Some("fɔɹ"),
        "from" => Some("fɹʌm"),
        "hello" => Some("həlo"),
        "i" => Some("aɪ"),
        "in" => Some("ɪn"),
        "is" => Some("ɪz"),
        "it" => Some("ɪt"),
        "of" => Some("ʌv"),
        "on" => Some("ɑn"),
        "that" => Some("ðæt"),
        "the" => Some("ðə"),
        "this" => Some("ðɪs"),
        "to" => Some("tu"),
        "was" => Some("wʌz"),
        "with" => Some("wɪθ"),
        "world" => Some("wɜld"),
        "you" => Some("ju"),
        _ => None,
    }
}

fn approximate_english_word(word: &str) -> String {
    let chars = word
        .trim_matches('\'')
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let rest = chars[i..].iter().collect::<String>();
        if rest.starts_with("tion") {
            out.push_str("ʃən");
            i += 4;
        } else if rest.starts_with("ough") {
            out.push_str("o");
            i += 4;
        } else if rest.starts_with("ch") {
            out.push_str("tʃ");
            i += 2;
        } else if rest.starts_with("sh") {
            out.push('ʃ');
            i += 2;
        } else if rest.starts_with("th") {
            out.push('θ');
            i += 2;
        } else if rest.starts_with("ph") {
            out.push('f');
            i += 2;
        } else if rest.starts_with("ng") {
            out.push('ŋ');
            i += 2;
        } else if rest.starts_with("qu") {
            out.push_str("kw");
            i += 2;
        } else if rest.starts_with("ee") || rest.starts_with("ea") {
            out.push('i');
            i += 2;
        } else if rest.starts_with("oo") {
            out.push('u');
            i += 2;
        } else if rest.starts_with("ow") || rest.starts_with("ou") {
            out.push_str("aʊ");
            i += 2;
        } else if rest.starts_with("er") || rest.starts_with("ir") || rest.starts_with("ur") {
            out.push('ɜ');
            i += 2;
        } else if rest.starts_with("ar") {
            out.push_str("ɑɹ");
            i += 2;
        } else if rest.starts_with("or") {
            out.push_str("ɔɹ");
            i += 2;
        } else if chars[i] == 'x' {
            out.push_str("ks");
            i += 1;
        } else {
            let phoneme = match chars[i] {
                'a' => 'æ',
                'b' => 'b',
                'c' => {
                    if matches!(chars.get(i + 1), Some('e' | 'i' | 'y')) {
                        's'
                    } else {
                        'k'
                    }
                }
                'd' => 'd',
                'e' => {
                    if i + 1 == chars.len() {
                        i += 1;
                        continue;
                    }
                    'ɛ'
                }
                'f' => 'f',
                'g' => 'ɡ',
                'h' => 'h',
                'i' => 'ɪ',
                'j' => 'd',
                'k' => 'k',
                'l' => 'l',
                'm' => 'm',
                'n' => 'n',
                'o' => 'ɑ',
                'p' => 'p',
                'q' => 'k',
                'r' => 'ɹ',
                's' => 's',
                't' => 't',
                'u' => 'ʌ',
                'v' => 'v',
                'w' => 'w',
                'y' => 'j',
                'z' => 'z',
                _ => 'ə',
            };
            out.push(phoneme);
            i += 1;
        }
    }
    if out.is_empty() {
        out.push('ə');
    }
    out
}

fn push_pronunciation(
    map: &PiperPhonemeMap,
    word: &str,
    pronunciation: &str,
    phonemes: &mut Vec<char>,
) -> Result<()> {
    for phoneme in pronunciation.chars() {
        if map.ids_for_phoneme(phoneme).is_some() {
            phonemes.push(phoneme);
        } else {
            return Err(PiperError::Unsupported(format!(
                "English fallback phoneme {phoneme:?} for word {word:?} is not available in this Piper voice"
            )));
        }
    }
    Ok(())
}

fn push_separator(map: &PiperPhonemeMap, phonemes: &mut Vec<char>) {
    if phonemes.last() != Some(&'_') {
        push_if_mapped(map, '_', phonemes);
    }
}

fn push_if_mapped(map: &PiperPhonemeMap, phoneme: char, phonemes: &mut Vec<char>) {
    if map.ids_for_phoneme(phoneme).is_some() {
        phonemes.push(phoneme);
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
    fn espeak_fallback_maps_common_english_words() {
        let mut config = test_config();
        config.phoneme_type = PiperPhonemeType::Espeak;
        config.phoneme_id_map.extend([
            ("^".to_string(), vec![0]),
            ("$".to_string(), vec![4]),
            ("_".to_string(), vec![5]),
            ("h".to_string(), vec![6]),
            ("ə".to_string(), vec![7]),
            ("l".to_string(), vec![8]),
            ("o".to_string(), vec![9]),
            ("w".to_string(), vec![10]),
            ("ɜ".to_string(), vec![11]),
            ("d".to_string(), vec![12]),
        ]);
        config.num_symbols = 13;

        let phonemes = text_to_phonemes(&config, "hello world").unwrap();

        assert_eq!(
            phonemes,
            vec!['^', 'h', 'ə', 'l', 'o', '_', 'w', 'ɜ', 'l', 'd', '$']
        );
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
