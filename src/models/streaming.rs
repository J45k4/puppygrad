pub trait TokenDecoder {
    type Error;

    fn decode_tokens(&self, token_ids: &[usize]) -> Result<String, Self::Error>;

    fn decode_token(&self, token_id: usize) -> Result<String, Self::Error> {
        self.decode_tokens(&[token_id])
    }
}

pub trait RawTokenDecoder {
    type Error;

    fn raw_token(&self, token_id: usize) -> Result<String, Self::Error>;
}

pub fn escape_raw_token(token: &str) -> String {
    let mut escaped = String::with_capacity(token.len());
    for ch in token.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{{{:x}}}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IncrementalTextStreamer {
    decoded: String,
}

impl IncrementalTextStreamer {
    pub fn new(decoded: String) -> Self {
        Self { decoded }
    }

    pub fn decoded(&self) -> &str {
        &self.decoded
    }

    pub fn stream_token<D, F, E>(
        &mut self,
        decoder: &D,
        output_ids: &[usize],
        token_id: usize,
        on_text: &mut F,
    ) -> Result<(), E>
    where
        D: TokenDecoder,
        F: FnMut(&str) -> Result<(), E>,
        E: From<D::Error>,
    {
        let token_text = decoder.decode_token(token_id)?;
        if is_incremental_decode_safe(&token_text) {
            on_text(&token_text)?;
            self.decoded.push_str(&token_text);
            return Ok(());
        }

        let next_decoded = decoder.decode_tokens(output_ids)?;
        if let Some(delta) = next_decoded.strip_prefix(self.decoded.as_str()) {
            on_text(delta)?;
        } else {
            on_text(&token_text)?;
        }
        self.decoded = next_decoded;
        Ok(())
    }
}

pub fn is_incremental_decode_safe(token_text: &str) -> bool {
    !token_text.is_empty() && !token_text.contains(char::REPLACEMENT_CHARACTER)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockDecoder;

    impl TokenDecoder for MockDecoder {
        type Error = std::convert::Infallible;

        fn decode_tokens(&self, token_ids: &[usize]) -> Result<String, Self::Error> {
            Ok(token_ids
                .iter()
                .map(|token_id| match token_id {
                    0 => "hello",
                    1 => " world",
                    2 => "",
                    _ => "\u{fffd}",
                })
                .collect())
        }
    }

    #[test]
    fn streams_safe_token_text_incrementally() -> Result<(), std::convert::Infallible> {
        let mut streamer = IncrementalTextStreamer::new("hello".to_string());
        let mut chunks = Vec::new();

        streamer.stream_token(&MockDecoder, &[0, 1], 1, &mut |text| {
            chunks.push(text.to_string());
            Ok::<(), std::convert::Infallible>(())
        })?;

        assert_eq!(chunks, [" world"]);
        assert_eq!(streamer.decoded(), "hello world");
        Ok(())
    }

    #[test]
    fn falls_back_to_full_decode_for_unsafe_token_text() -> Result<(), std::convert::Infallible> {
        let mut streamer = IncrementalTextStreamer::new("hello".to_string());
        let mut chunks = Vec::new();

        streamer.stream_token(&MockDecoder, &[0, 1, 2], 2, &mut |text| {
            chunks.push(text.to_string());
            Ok::<(), std::convert::Infallible>(())
        })?;

        assert_eq!(chunks, [" world"]);
        assert_eq!(streamer.decoded(), "hello world");
        Ok(())
    }

    #[test]
    fn incremental_decode_rejects_empty_and_replacement_text() {
        assert!(is_incremental_decode_safe(" hello"));
        assert!(!is_incremental_decode_safe(""));
        assert!(!is_incremental_decode_safe("\u{fffd}"));
        assert!(!is_incremental_decode_safe("a\u{fffd}"));
    }

    #[test]
    fn escapes_raw_tokens_for_line_streams() {
        assert_eq!(escape_raw_token("<|en|>"), "<|en|>");
        assert_eq!(escape_raw_token("ĠI"), "ĠI");
        assert_eq!(escape_raw_token("a\tb\nc"), "a\\tb\\nc");
    }
}
