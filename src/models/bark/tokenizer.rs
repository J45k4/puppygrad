use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::{BarkError, Result};

const WORDPIECE_MAX_INPUT_CHARS_PER_WORD: usize = 100;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkTokenizer {
    vocab: HashMap<String, usize>,
    cls_token: String,
    sep_token: String,
    unk_token: String,
    pad_token: String,
    max_input_tokens: usize,
    do_lower_case: bool,
}

impl BarkTokenizer {
    pub fn from_model_files(
        vocab_path: impl AsRef<Path>,
        tokenizer_config_path: impl AsRef<Path>,
        special_tokens_map_path: impl AsRef<Path>,
        max_input_tokens: usize,
    ) -> Result<Self> {
        let metadata =
            BarkTokenizerMetadata::from_files(tokenizer_config_path, special_tokens_map_path)?;
        Self::from_vocab_file_with_metadata(vocab_path, max_input_tokens, metadata)
    }

    pub fn from_vocab_file(path: impl AsRef<Path>, max_input_tokens: usize) -> Result<Self> {
        Self::from_vocab_file_with_metadata(
            path,
            max_input_tokens,
            BarkTokenizerMetadata::default(),
        )
    }

    fn from_vocab_file_with_metadata(
        path: impl AsRef<Path>,
        max_input_tokens: usize,
        metadata: BarkTokenizerMetadata,
    ) -> Result<Self> {
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
        Self::from_vocab_with_metadata(vocab, max_input_tokens, metadata)
    }

    pub fn from_vocab(vocab: HashMap<String, usize>, max_input_tokens: usize) -> Result<Self> {
        Self::from_vocab_with_metadata(vocab, max_input_tokens, BarkTokenizerMetadata::default())
    }

    fn from_vocab_with_metadata(
        vocab: HashMap<String, usize>,
        max_input_tokens: usize,
        metadata: BarkTokenizerMetadata,
    ) -> Result<Self> {
        let tokenizer = Self {
            vocab,
            cls_token: metadata.cls_token,
            sep_token: metadata.sep_token,
            unk_token: metadata.unk_token,
            pad_token: metadata.pad_token,
            max_input_tokens,
            do_lower_case: metadata.do_lower_case,
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
        let mut tokens = self.wordpiece_tokens(text)?;
        tokens.insert(0, self.cls_token.clone());
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

    pub fn encode_for_processor(&self, text: &str) -> Result<BarkTextEncoding> {
        if text.trim().is_empty() {
            return Err(BarkError::InvalidInput(
                "text must not be empty".to_string(),
            ));
        }
        let mut tokens = self.wordpiece_tokens(text)?;
        tokens.truncate(self.max_input_tokens);

        let mut ids = Vec::with_capacity(self.max_input_tokens);
        for token in &tokens {
            ids.push(*self.vocab.get(token).ok_or_else(|| {
                BarkError::Tokenizer(format!("token {token:?} missing from vocab"))
            })?);
        }
        let pad_id = *self.vocab.get(&self.pad_token).ok_or_else(|| {
            BarkError::Tokenizer(format!("token {:?} missing from vocab", self.pad_token))
        })?;
        ids.resize(self.max_input_tokens, pad_id);
        tokens.resize(self.max_input_tokens, self.pad_token.clone());
        Ok(BarkTextEncoding { tokens, ids })
    }

    pub fn encode_for_semantic_model(
        &self,
        text: &str,
        offset: usize,
        text_pad_token: usize,
    ) -> Result<Vec<usize>> {
        let encoding = self.encode_for_processor(text)?;
        let pad_token = self.pad_token.clone();
        Ok(encoding
            .tokens
            .into_iter()
            .zip(encoding.ids)
            .map(|(token, token_id)| {
                if token == pad_token {
                    text_pad_token
                } else {
                    token_id + offset
                }
            })
            .collect())
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    pub fn do_lower_case(&self) -> bool {
        self.do_lower_case
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
        if chars.len() > WORDPIECE_MAX_INPUT_CHARS_PER_WORD {
            return vec![self.unk_token.clone()];
        }
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

    fn wordpiece_tokens(&self, text: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        for piece in basic_tokenize(text, self.do_lower_case) {
            let pieces = self.wordpiece_tokenize(&piece);
            tokens.extend(pieces);
        }
        Ok(tokens)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkTextEncoding {
    pub tokens: Vec<String>,
    pub ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct BarkTokenizerJson {
    #[serde(default)]
    do_lower_case: Option<bool>,
    #[serde(default)]
    cls_token: Option<TokenSpec>,
    #[serde(default)]
    sep_token: Option<TokenSpec>,
    #[serde(default)]
    unk_token: Option<TokenSpec>,
    #[serde(default)]
    pad_token: Option<TokenSpec>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct BarkSpecialTokensJson {
    #[serde(default)]
    cls_token: Option<TokenSpec>,
    #[serde(default)]
    sep_token: Option<TokenSpec>,
    #[serde(default)]
    unk_token: Option<TokenSpec>,
    #[serde(default)]
    pad_token: Option<TokenSpec>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
enum TokenSpec {
    String(String),
    AddedToken { content: String },
}

impl TokenSpec {
    fn into_content(self) -> String {
        match self {
            TokenSpec::String(content) | TokenSpec::AddedToken { content } => content,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BarkTokenizerMetadata {
    cls_token: String,
    sep_token: String,
    unk_token: String,
    pad_token: String,
    do_lower_case: bool,
}

impl BarkTokenizerMetadata {
    fn from_files(
        tokenizer_config_path: impl AsRef<Path>,
        special_tokens_map_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let tokenizer_config_path = tokenizer_config_path.as_ref();
        let tokenizer_config: BarkTokenizerJson =
            serde_json::from_str(&fs::read_to_string(tokenizer_config_path).map_err(|err| {
                BarkError::Tokenizer(format!(
                    "failed to read tokenizer config {}: {err}",
                    tokenizer_config_path.display()
                ))
            })?)
            .map_err(|err| {
                BarkError::Tokenizer(format!(
                    "failed to parse tokenizer config {}: {err}",
                    tokenizer_config_path.display()
                ))
            })?;

        let special_tokens_map_path = special_tokens_map_path.as_ref();
        let special_tokens: BarkSpecialTokensJson =
            serde_json::from_str(&fs::read_to_string(special_tokens_map_path).map_err(|err| {
                BarkError::Tokenizer(format!(
                    "failed to read special tokens map {}: {err}",
                    special_tokens_map_path.display()
                ))
            })?)
            .map_err(|err| {
                BarkError::Tokenizer(format!(
                    "failed to parse special tokens map {}: {err}",
                    special_tokens_map_path.display()
                ))
            })?;

        let mut metadata = Self {
            do_lower_case: tokenizer_config.do_lower_case.unwrap_or(false),
            cls_token: token_or_default(tokenizer_config.cls_token, "[CLS]"),
            sep_token: token_or_default(tokenizer_config.sep_token, "[SEP]"),
            unk_token: token_or_default(tokenizer_config.unk_token, "[UNK]"),
            pad_token: token_or_default(tokenizer_config.pad_token, "[PAD]"),
        };
        if let Some(token) = special_tokens.cls_token {
            metadata.cls_token = token.into_content();
        }
        if let Some(token) = special_tokens.sep_token {
            metadata.sep_token = token.into_content();
        }
        if let Some(token) = special_tokens.unk_token {
            metadata.unk_token = token.into_content();
        }
        if let Some(token) = special_tokens.pad_token {
            metadata.pad_token = token.into_content();
        }
        Ok(metadata)
    }
}

impl Default for BarkTokenizerMetadata {
    fn default() -> Self {
        Self {
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
            do_lower_case: false,
        }
    }
}

fn token_or_default(token: Option<TokenSpec>, default: &str) -> String {
    token
        .map(TokenSpec::into_content)
        .unwrap_or_else(|| default.to_string())
}

fn basic_tokenize(text: &str, do_lower_case: bool) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in clean_and_split_cjk_chars(text) {
        if ch.is_whitespace() {
            push_current(&mut tokens, &mut current, do_lower_case);
        } else if is_punctuation(ch) {
            push_current(&mut tokens, &mut current, do_lower_case);
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    push_current(&mut tokens, &mut current, do_lower_case);
    tokens
}

fn clean_and_split_cjk_chars(text: &str) -> Vec<char> {
    let mut chars = Vec::new();
    for ch in text.chars() {
        if ch == '\0' || ch == '\u{fffd}' || is_control(ch) {
            continue;
        }
        if is_whitespace(ch) {
            chars.push(' ');
        } else if is_cjk_char(ch) {
            chars.push(' ');
            chars.push(ch);
            chars.push(' ');
        } else {
            chars.push(ch);
        }
    }
    chars
}

fn normalize_basic_token(token: &str, do_lower_case: bool) -> String {
    if do_lower_case {
        lowercase_and_strip_accents(token)
    } else {
        token.to_string()
    }
}

fn push_current(tokens: &mut Vec<String>, current: &mut String, do_lower_case: bool) {
    if !current.is_empty() {
        let token = normalize_basic_token(current, do_lower_case);
        current.clear();
        if !token.is_empty() {
            tokens.push(token);
        }
    }
}

fn lowercase_and_strip_accents(token: &str) -> String {
    let mut normalized = String::with_capacity(token.len());
    for ch in token.chars().flat_map(char::to_lowercase) {
        if is_combining_mark(ch) {
            continue;
        }
        match latin_accent_base(ch) {
            Some(base) => normalized.push_str(base),
            None => normalized.push(ch),
        }
    }
    normalized
}

fn is_whitespace(ch: char) -> bool {
    ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch.is_whitespace()
}

fn is_control(ch: char) -> bool {
    !matches!(ch, '\t' | '\n' | '\r') && ch.is_control()
}

fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{205e}' | '\u{3001}'..='\u{303f}'
        )
}

fn is_combining_mark(ch: char) -> bool {
    matches!(
        ch,
        '\u{0300}'..='\u{036f}'
            | '\u{1ab0}'..='\u{1aff}'
            | '\u{1dc0}'..='\u{1dff}'
            | '\u{20d0}'..='\u{20ff}'
            | '\u{fe20}'..='\u{fe2f}'
    )
}

fn is_cjk_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{4e00}'..='\u{9fff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{20000}'..='\u{2a6df}'
            | '\u{2a700}'..='\u{2b73f}'
            | '\u{2b740}'..='\u{2b81f}'
            | '\u{2b820}'..='\u{2ceaf}'
            | '\u{f900}'..='\u{faff}'
            | '\u{2f800}'..='\u{2fa1f}'
    )
}

fn latin_accent_base(ch: char) -> Option<&'static str> {
    Some(match ch {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a",
        'æ' => "ae",
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'ď' | 'đ' => "d",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => "e",
        'ƒ' => "f",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'ĥ' | 'ħ' => "h",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => "i",
        'ĵ' => "j",
        'ķ' => "k",
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l",
        'ñ' | 'ń' | 'ņ' | 'ň' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => "o",
        'œ' => "oe",
        'ŕ' | 'ŗ' | 'ř' => "r",
        'ś' | 'ŝ' | 'ş' | 'š' => "s",
        'ß' => "ss",
        'ţ' | 'ť' | 'ŧ' => "t",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => "u",
        'ŵ' => "w",
        'ý' | 'ÿ' | 'ŷ' => "y",
        'ź' | 'ż' | 'ž' => "z",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{load_bark_generation_config, BarkAssetPaths};
    use super::*;

    const TOKENIZER_REFERENCE_FIXTURE: &str = "tests/data/bark/tokenizer/reference.json";

    #[derive(Debug, Deserialize)]
    struct TokenizerReferenceFixture {
        metadata: TokenizerReferenceMetadata,
        cases: Vec<TokenizerReferenceCase>,
    }

    #[derive(Debug, Deserialize)]
    struct TokenizerReferenceMetadata {
        model_dir: String,
        case_count: usize,
    }

    #[derive(Debug, Deserialize)]
    struct TokenizerReferenceCase {
        label: String,
        text: String,
        input_ids: Vec<usize>,
        tokens: Vec<String>,
        semantic_input_ids: Vec<usize>,
    }

    fn tiny_vocab_tokens() -> [&'static str; 12] {
        [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "!", "play", "##ing", "cafe",
            "中", "文",
        ]
    }

    fn tiny_tokenizer() -> BarkTokenizer {
        let mut vocab = HashMap::new();
        for (idx, token) in tiny_vocab_tokens().into_iter().enumerate() {
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
            .encode_for_semantic_model("hello world", 10_048, 129_595)
            .unwrap();

        assert_eq!(
            ids,
            vec![10_052, 10_053, 129_595, 129_595, 129_595, 129_595, 129_595, 129_595]
        );
    }

    #[test]
    fn transformers_tokenizer_reference_fixtures_match_when_available() -> Result<()> {
        let path = Path::new(TOKENIZER_REFERENCE_FIXTURE);
        if !path.exists() {
            return Ok(());
        }
        let raw = fs::read_to_string(path)
            .map_err(|err| BarkError::Asset(format!("failed to read {}: {err}", path.display())))?;
        let fixture: TokenizerReferenceFixture = serde_json::from_str(&raw).map_err(|err| {
            BarkError::Asset(format!("failed to parse {}: {err}", path.display()))
        })?;
        assert_eq!(fixture.metadata.case_count, fixture.cases.len());

        let paths = BarkAssetPaths::new(&fixture.metadata.model_dir);
        if !paths.vocab.exists()
            || !paths.tokenizer_config.exists()
            || !paths.special_tokens_map.exists()
            || !paths.generation_config.exists()
        {
            return Ok(());
        }
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let tokenizer = BarkTokenizer::from_model_files(
            &paths.vocab,
            &paths.tokenizer_config,
            &paths.special_tokens_map,
            generation_config.semantic_config.max_input_semantic_length,
        )?;

        for case in fixture.cases {
            let encoding = tokenizer.encode_for_processor(&case.text)?;
            assert_eq!(encoding.ids, case.input_ids, "{}", case.label);
            assert_eq!(encoding.tokens, case.tokens, "{}", case.label);
            let semantic_ids = tokenizer.encode_for_semantic_model(
                &case.text,
                generation_config.semantic_config.text_encoding_offset,
                generation_config.semantic_config.text_pad_token,
            )?;
            assert_eq!(semantic_ids, case.semantic_input_ids, "{}", case.label);
        }
        Ok(())
    }

    #[test]
    fn local_wordpiece_matches_tokenizers_bert_pipeline_for_bark_cases() {
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::normalizers::bert::BertNormalizer;
        use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
        use tokenizers::processors::bert::BertProcessing;
        use tokenizers::Tokenizer;

        let dir = std::env::temp_dir().join(format!(
            "puppygrad-bark-hf-tokenizer-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = dir.join("vocab.txt");
        std::fs::write(&vocab, tiny_vocab_tokens().join("\n")).unwrap();

        let metadata = BarkTokenizerMetadata {
            do_lower_case: true,
            ..BarkTokenizerMetadata::default()
        };
        let local =
            BarkTokenizer::from_vocab_with_metadata(tiny_tokenizer().vocab, 16, metadata).unwrap();
        let local_encoding = local.encode("CAFÉ! 中文 playing unseen").unwrap();

        let mut hf = Tokenizer::new(
            WordPiece::builder()
                .files(vocab.to_string_lossy().into_owned())
                .unk_token("[UNK]".to_string())
                .build()
                .unwrap(),
        );
        hf.with_normalizer(Some(BertNormalizer::new(true, true, Some(true), true)))
            .unwrap()
            .with_pre_tokenizer(Some(BertPreTokenizer))
            .with_post_processor(Some(BertProcessing::new(
                ("[SEP]".to_string(), 3),
                ("[CLS]".to_string(), 2),
            )));
        let hf_encoding = hf.encode("CAFÉ! 中文 playing unseen", true).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(local_encoding.tokens, hf_encoding.get_tokens());
        assert_eq!(
            local_encoding.ids,
            hf_encoding
                .get_ids()
                .iter()
                .map(|id| *id as usize)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn overlong_wordpiece_token_falls_back_to_unk_like_bert() {
        let mut vocab = HashMap::new();
        for (idx, token) in tiny_vocab_tokens().into_iter().enumerate() {
            vocab.insert(token.to_string(), idx);
        }
        vocab.insert("a".repeat(101), vocab.len());
        let tokenizer = BarkTokenizer::from_vocab(vocab, 8).unwrap();

        let encoding = tokenizer.encode(&"a".repeat(101)).unwrap();

        assert_eq!(encoding.tokens, ["[CLS]", "[UNK]", "[SEP]"]);
    }

    #[test]
    fn truncates_and_keeps_sep() {
        let tokenizer = tiny_tokenizer();

        let encoding = tokenizer.encode("hello world hello world").unwrap();

        assert_eq!(encoding.tokens.len(), 6);
        assert_eq!(encoding.tokens.last().unwrap(), "[SEP]");
    }

    #[test]
    fn loads_special_tokens_and_lowercase_from_metadata_files() {
        let dir =
            std::env::temp_dir().join(format!("puppygrad-bark-tokenizer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = dir.join("vocab.txt");
        let tokenizer_config = dir.join("tokenizer_config.json");
        let special_tokens = dir.join("special_tokens_map.json");
        std::fs::write(&vocab, "[PAD]\n[UNK]\n<s>\n</s>\nhello\nworld\n!\n").unwrap();
        std::fs::write(
            &tokenizer_config,
            r#"{"do_lower_case": true, "cls_token": "[CLS]", "sep_token": "[SEP]"}"#,
        )
        .unwrap();
        std::fs::write(
            &special_tokens,
            r#"{"cls_token": {"content": "<s>"}, "sep_token": "</s>", "unk_token": "[UNK]", "pad_token": "[PAD]"}"#,
        )
        .unwrap();

        let tokenizer =
            BarkTokenizer::from_model_files(&vocab, &tokenizer_config, &special_tokens, 8).unwrap();
        let encoding = tokenizer.encode("Hello WORLD!").unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(tokenizer.do_lower_case());
        assert_eq!(encoding.tokens, ["<s>", "hello", "world", "!", "</s>"]);
    }

    #[test]
    fn basic_tokenizer_matches_bert_style_lowercase_accents_punctuation_and_cjk() {
        let metadata = BarkTokenizerMetadata {
            do_lower_case: true,
            ..BarkTokenizerMetadata::default()
        };
        let tokenizer =
            BarkTokenizer::from_vocab_with_metadata(tiny_tokenizer().vocab, 16, metadata).unwrap();

        let encoding = tokenizer.encode("CAFÉ! 中文").unwrap();

        assert_eq!(encoding.tokens, ["[CLS]", "cafe", "!", "中", "文", "[SEP]"]);
    }

    #[test]
    fn unknown_wordpiece_falls_back_to_unk() {
        let tokenizer = tiny_tokenizer();

        let encoding = tokenizer.encode("unseen").unwrap();

        assert_eq!(encoding.tokens, ["[CLS]", "[UNK]", "[SEP]"]);
    }
}
