use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::{BarkError, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkTokenizer {
    vocab: HashMap<String, usize>,
    cls_token: String,
    sep_token: String,
    unk_token: String,
    pad_token: String,
    max_input_tokens: usize,
}

impl BarkTokenizer {
    pub fn from_vocab_file(path: impl AsRef<Path>, max_input_tokens: usize) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read_to_string(path).map_err(|err| {
            BarkError::Tokenizer(format!("failed to read vocab {}: {err}", path.display()))
        })?;
        let mut vocab = HashMap::new();
        for (idx, token) in data.lines().enumerate() {
            let token = token.trim_end_matches('\r');
            if token.is_empty() {
                continue;
            }
            if vocab.insert(token.to_string(), idx).is_some() {
                return Err(BarkError::Tokenizer(format!(
                    "duplicate token {token:?} in {}",
                    path.display()
                )));
            }
        }
        Self::from_vocab(vocab, max_input_tokens)
    }

    pub fn from_vocab(vocab: HashMap<String, usize>, max_input_tokens: usize) -> Result<Self> {
        let tokenizer = Self {
            vocab,
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
            max_input_tokens,
        };
        tokenizer.validate()?;
        Ok(tokenizer)
    }

    pub fn encode(&self, text: &str) -> Result<BarkTextEncoding> {
        if text.trim().is_empty() {
            return Err(BarkError::InvalidInput(
                "text must not be empty".to_string(),
            ));
        }
        let mut tokens = Vec::new();
        tokens.push(self.cls_token.clone());
        for piece in basic_tokenize(text) {
            let pieces = self.wordpiece_tokenize(&piece);
            tokens.extend(pieces);
        }
        tokens.push(self.sep_token.clone());
        if tokens.len() > self.max_input_tokens {
            tokens.truncate(self.max_input_tokens);
            let sep = self.sep_token.clone();
            if let Some(last) = tokens.last_mut() {
                *last = sep;
            }
        }

        let mut ids = Vec::with_capacity(tokens.len());
        for token in &tokens {
            ids.push(*self.vocab.get(token).ok_or_else(|| {
                BarkError::Tokenizer(format!("token {token:?} missing from vocab"))
            })?);
        }
        Ok(BarkTextEncoding { tokens, ids })
    }

    pub fn encode_for_semantic_model(&self, text: &str, offset: usize) -> Result<Vec<usize>> {
        let encoding = self.encode(text)?;
        Ok(encoding
            .ids
            .into_iter()
            .map(|token_id| token_id + offset)
            .collect())
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    fn validate(&self) -> Result<()> {
        if self.max_input_tokens < 2 {
            return Err(BarkError::Tokenizer(
                "max_input_tokens must allow [CLS] and [SEP]".to_string(),
            ));
        }
        for token in [
            &self.cls_token,
            &self.sep_token,
            &self.unk_token,
            &self.pad_token,
        ] {
            if !self.vocab.contains_key(token) {
                return Err(BarkError::Tokenizer(format!(
                    "required token {token:?} is missing from vocab"
                )));
            }
        }
        Ok(())
    }

    fn wordpiece_tokenize(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }
        let chars = token.chars().collect::<Vec<_>>();
        let mut start = 0;
        let mut pieces = Vec::new();
        while start < chars.len() {
            let mut end = chars.len();
            let mut current = None;
            while start < end {
                let mut piece = chars[start..end].iter().collect::<String>();
                if start > 0 {
                    piece.insert_str(0, "##");
                }
                if self.vocab.contains_key(&piece) {
                    current = Some(piece);
                    break;
                }
                end -= 1;
            }
            match current {
                Some(piece) => {
                    pieces.push(piece);
                    start = end;
                }
                None => return vec![self.unk_token.clone()],
            }
        }
        pieces
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkTextEncoding {
    pub tokens: Vec<String>,
    pub ids: Vec<usize>,
}

fn basic_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            push_current(&mut tokens, &mut current);
        } else if is_punctuation(ch) {
            push_current(&mut tokens, &mut current);
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    push_current(&mut tokens, &mut current);
    tokens
}

fn push_current(tokens: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{205e}' | '\u{3001}'..='\u{303f}'
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> BarkTokenizer {
        let mut vocab = HashMap::new();
        for (idx, token) in [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "!", "play", "##ing",
        ]
        .into_iter()
        .enumerate()
        {
            vocab.insert(token.to_string(), idx);
        }
        BarkTokenizer::from_vocab(vocab, 8).unwrap()
    }

    #[test]
    fn encodes_basic_wordpiece_text() {
        let tokenizer = tiny_tokenizer();

        let encoding = tokenizer.encode("hello playing!").unwrap();

        assert_eq!(
            encoding.tokens,
            ["[CLS]", "hello", "play", "##ing", "!", "[SEP]"]
        );
    }

    #[test]
    fn applies_semantic_offset() {
        let tokenizer = tiny_tokenizer();

        let ids = tokenizer
            .encode_for_semantic_model("hello world", 10_048)
            .unwrap();

        assert_eq!(ids, vec![10_050, 10_052, 10_053, 10_051]);
    }

    #[test]
    fn truncates_and_keeps_sep() {
        let tokenizer = tiny_tokenizer();

        let encoding = tokenizer.encode("hello world hello world").unwrap();

        assert_eq!(encoding.tokens.len(), 6);
        assert_eq!(encoding.tokens.last().unwrap(), "[SEP]");
    }
}
