use std::path::Path;

use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::{unicode::NFC, utils::Lowercase, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;

use super::{
    layer_norm_last_dim, linear2d, scaled_dot_product_attention, ClipTextConfig, ClipTextWeights,
    Result, SdTensor, StableDiffusionError,
};

pub const SD1_CLIP_MAX_TOKENS: usize = 77;

#[derive(Clone)]
pub struct StableDiffusionTokenizer {
    tokenizer: Tokenizer,
    special_tokens: StableDiffusionTokenizerSpecialTokens,
    max_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableDiffusionTokenizerSpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableDiffusionTokenizedPrompt {
    pub token_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableDiffusionConditioningTokens {
    pub prompt: StableDiffusionTokenizedPrompt,
    pub negative_prompt: StableDiffusionTokenizedPrompt,
}

#[derive(Clone, Debug)]
pub struct ClipTextEncoder {
    config: ClipTextConfig,
    weights: ClipTextWeights,
}

impl StableDiffusionTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref()).map_err(|err| {
            StableDiffusionError::Asset(format!(
                "failed to load Stable Diffusion tokenizer {}: {err}",
                path.as_ref().display()
            ))
        })?;
        Self::from_tokenizer(tokenizer)
    }

    pub fn from_diffusers_files(
        tokenizer_json: impl AsRef<Path>,
        vocab_json: impl AsRef<Path>,
        merges_txt: impl AsRef<Path>,
    ) -> Result<Self> {
        let tokenizer_json = tokenizer_json.as_ref();
        if tokenizer_json.is_file() {
            return Self::from_file(tokenizer_json);
        }

        let vocab_json = vocab_json.as_ref();
        let merges_txt = merges_txt.as_ref();
        let model = BPE::from_file(
            vocab_json.to_str().ok_or_else(|| {
                StableDiffusionError::Asset(format!(
                    "tokenizer vocab path is not valid UTF-8: {}",
                    vocab_json.display()
                ))
            })?,
            merges_txt.to_str().ok_or_else(|| {
                StableDiffusionError::Asset(format!(
                    "tokenizer merges path is not valid UTF-8: {}",
                    merges_txt.display()
                ))
            })?,
        )
        .unk_token("<|endoftext|>".to_string())
        .end_of_word_suffix("</w>".to_string())
        .build()
        .map_err(|err| {
            StableDiffusionError::Asset(format!(
                "failed to load Stable Diffusion tokenizer from {} and {}: {err}",
                vocab_json.display(),
                merges_txt.display()
            ))
        })?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer
            .with_normalizer(Some(Sequence::new(vec![NFC.into(), Lowercase.into()])))
            .map_err(|err| {
                StableDiffusionError::Asset(format!(
                    "failed to configure Stable Diffusion tokenizer normalizer: {err}"
                ))
            })?;
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        Self::from_tokenizer(tokenizer)
    }

    pub fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let special_tokens = StableDiffusionTokenizerSpecialTokens::from_tokenizer(&tokenizer)?;
        Ok(Self {
            tokenizer,
            special_tokens,
            max_tokens: SD1_CLIP_MAX_TOKENS,
        })
    }

    pub fn special_tokens(&self) -> &StableDiffusionTokenizerSpecialTokens {
        &self.special_tokens
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<StableDiffusionTokenizedPrompt> {
        self.encode_prompt_with_max_len(prompt, self.max_tokens)
    }

    pub fn encode_conditioning(
        &self,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<StableDiffusionConditioningTokens> {
        Ok(StableDiffusionConditioningTokens {
            prompt: self.encode_prompt(prompt)?,
            negative_prompt: self.encode_prompt(negative_prompt)?,
        })
    }

    fn encode_prompt_with_max_len(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<StableDiffusionTokenizedPrompt> {
        if max_tokens < 2 {
            return Err(StableDiffusionError::InvalidInput(
                "CLIP max token length must leave room for BOS and EOS".to_string(),
            ));
        }
        let encoding = self.tokenizer.encode(prompt, false).map_err(|err| {
            StableDiffusionError::Asset(format!("failed to encode Stable Diffusion prompt: {err}"))
        })?;
        let raw_ids = encoding.get_ids();
        let content_limit = max_tokens - 2;
        let truncated = raw_ids.len() > content_limit;

        let mut token_ids = Vec::with_capacity(max_tokens);
        token_ids.push(self.special_tokens.bos);
        token_ids.extend(raw_ids.iter().take(content_limit).copied());
        token_ids.push(self.special_tokens.eos);

        let mut attention_mask = vec![1; token_ids.len()];
        while token_ids.len() < max_tokens {
            token_ids.push(self.special_tokens.pad);
            attention_mask.push(0);
        }

        Ok(StableDiffusionTokenizedPrompt {
            token_ids,
            attention_mask,
            truncated,
        })
    }
}

impl ClipTextEncoder {
    pub fn new(config: ClipTextConfig, weights: ClipTextWeights) -> Result<Self> {
        if weights.layers.len() != config.num_hidden_layers {
            return Err(StableDiffusionError::Asset(format!(
                "CLIP text weights have {} layers, config expects {}",
                weights.layers.len(),
                config.num_hidden_layers
            )));
        }
        Ok(Self { config, weights })
    }

    pub fn encode_token_ids(&self, token_ids: &[u32]) -> Result<SdTensor> {
        if token_ids.is_empty() {
            return Err(StableDiffusionError::InvalidInput(
                "CLIP text encoder requires at least one token".to_string(),
            ));
        }
        if token_ids.len() > self.config.max_position_embeddings {
            return Err(StableDiffusionError::InvalidInput(format!(
                "CLIP token sequence length {} exceeds max_position_embeddings {}",
                token_ids.len(),
                self.config.max_position_embeddings
            )));
        }

        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let mut hidden = vec![0.0; seq_len * hidden_size];
        for (position, token_id) in token_ids.iter().copied().enumerate() {
            let token_id = usize::try_from(token_id).map_err(|_| {
                StableDiffusionError::InvalidInput(format!("CLIP token id {token_id} is invalid"))
            })?;
            if token_id >= self.config.vocab_size {
                return Err(StableDiffusionError::InvalidInput(format!(
                    "CLIP token id {token_id} exceeds vocab_size {}",
                    self.config.vocab_size
                )));
            }
            for dim in 0..hidden_size {
                hidden[position * hidden_size + dim] = self.weights.token_embedding
                    [token_id * hidden_size + dim]
                    + self.weights.position_embedding[position * hidden_size + dim];
            }
        }
        let mut hidden = SdTensor::new([seq_len, hidden_size], hidden)?;
        let causal_mask = causal_attention_mask(seq_len)?;

        for layer in &self.weights.layers {
            let norm1 = layer_norm_last_dim(
                &hidden,
                &layer.layer_norm1_weight,
                &layer.layer_norm1_bias,
                self.config.layer_norm_eps,
            )?;
            let q = dense(
                &norm1,
                &layer.self_attn.q_proj_weight,
                &layer.self_attn.q_proj_bias,
                hidden_size,
            )?;
            let k = dense(
                &norm1,
                &layer.self_attn.k_proj_weight,
                &layer.self_attn.k_proj_bias,
                hidden_size,
            )?;
            let v = dense(
                &norm1,
                &layer.self_attn.v_proj_weight,
                &layer.self_attn.v_proj_bias,
                hidden_size,
            )?;
            let q = split_heads(&q, self.config.num_attention_heads)?;
            let k = split_heads(&k, self.config.num_attention_heads)?;
            let v = split_heads(&v, self.config.num_attention_heads)?;
            let attn = scaled_dot_product_attention(&q, &k, &v, Some(&causal_mask))?;
            let attn = merge_heads(&attn)?;
            let attn = dense(
                &attn,
                &layer.self_attn.out_proj_weight,
                &layer.self_attn.out_proj_bias,
                hidden_size,
            )?;
            hidden = hidden.add(&attn)?;

            let norm2 = layer_norm_last_dim(
                &hidden,
                &layer.layer_norm2_weight,
                &layer.layer_norm2_bias,
                self.config.layer_norm_eps,
            )?;
            let mlp = dense(
                &norm2,
                &layer.mlp_fc1_weight,
                &layer.mlp_fc1_bias,
                self.config.intermediate_size,
            )?;
            let mlp = clip_activation(&mlp, self.config.hidden_act.as_str())?;
            let mlp = dense(
                &mlp,
                &layer.mlp_fc2_weight,
                &layer.mlp_fc2_bias,
                hidden_size,
            )?;
            hidden = hidden.add(&mlp)?;
        }

        let hidden = layer_norm_last_dim(
            &hidden,
            &self.weights.final_layer_norm_weight,
            &self.weights.final_layer_norm_bias,
            self.config.layer_norm_eps,
        )?;
        SdTensor::new([1, seq_len, hidden_size], hidden.data().to_vec())
    }
}

impl StableDiffusionTokenizerSpecialTokens {
    fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        let bos = token_id(tokenizer, "<|startoftext|>")?;
        let eos = token_id(tokenizer, "<|endoftext|>")?;
        let pad = tokenizer
            .get_padding()
            .map(|padding| padding.pad_id)
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(eos);
        Ok(Self { bos, eos, pad })
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        StableDiffusionError::Asset(format!(
            "Stable Diffusion tokenizer is missing required token {token}"
        ))
    })
}

pub fn validate_clip_token_ids(token_ids: &[u32]) -> Result<()> {
    if token_ids.len() != SD1_CLIP_MAX_TOKENS {
        return Err(StableDiffusionError::InvalidInput(format!(
            "SD 1.x CLIP token ids must have length {SD1_CLIP_MAX_TOKENS}, got {}",
            token_ids.len()
        )));
    }
    Ok(())
}

fn dense(input: &SdTensor, weight: &[f32], bias: &[f32], out_features: usize) -> Result<SdTensor> {
    if input.rank() != 2 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "CLIP dense input must be rank-2, got {:?}",
            input.shape()
        )));
    }
    let in_features = input.shape()[1];
    if weight.len() != in_features * out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "CLIP dense weight expected {} values, got {}",
            in_features * out_features,
            weight.len()
        )));
    }
    if bias.len() != out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "CLIP dense bias expected {out_features} values, got {}",
            bias.len()
        )));
    }

    linear2d(input, weight, Some(bias), in_features, out_features)
}

fn clip_activation(input: &SdTensor, hidden_act: &str) -> Result<SdTensor> {
    match hidden_act {
        "" | "quick_gelu" => SdTensor::new(
            input.shape().to_vec(),
            input
                .data()
                .iter()
                .map(|value| {
                    let x = *value;
                    x / (1.0 + (-1.702 * x).exp())
                })
                .collect(),
        ),
        "gelu" => input.gelu(),
        other => Err(StableDiffusionError::Unsupported(format!(
            "CLIP activation {other} is not supported"
        ))),
    }
}

fn split_heads(input: &SdTensor, heads: usize) -> Result<SdTensor> {
    if input.rank() != 2 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "CLIP split_heads requires rank-2 input, got {:?}",
            input.shape()
        )));
    }
    let seq_len = input.shape()[0];
    let hidden_size = input.shape()[1];
    if heads == 0 || hidden_size % heads != 0 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "CLIP hidden size {hidden_size} must be divisible by attention heads {heads}"
        )));
    }
    let head_dim = hidden_size / heads;
    let mut out = vec![0.0; heads * seq_len * head_dim];
    for seq in 0..seq_len {
        for head in 0..heads {
            for dim in 0..head_dim {
                out[(head * seq_len + seq) * head_dim + dim] =
                    input.data()[seq * hidden_size + head * head_dim + dim];
            }
        }
    }
    SdTensor::new([1, heads, seq_len, head_dim], out)
}

fn merge_heads(input: &SdTensor) -> Result<SdTensor> {
    if input.rank() != 4 || input.shape()[0] != 1 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "CLIP merge_heads expects [1, heads, seq, dim], got {:?}",
            input.shape()
        )));
    }
    let heads = input.shape()[1];
    let seq_len = input.shape()[2];
    let head_dim = input.shape()[3];
    let hidden_size = heads * head_dim;
    let mut out = vec![0.0; seq_len * hidden_size];
    for seq in 0..seq_len {
        for head in 0..heads {
            for dim in 0..head_dim {
                out[seq * hidden_size + head * head_dim + dim] =
                    input.data()[(head * seq_len + seq) * head_dim + dim];
            }
        }
    }
    SdTensor::new([seq_len, hidden_size], out)
}

fn causal_attention_mask(seq_len: usize) -> Result<SdTensor> {
    let mut data = vec![0.0; seq_len * seq_len];
    for query in 0..seq_len {
        for key in query + 1..seq_len {
            data[query * seq_len + key] = -10_000.0;
        }
    }
    SdTensor::new([1, 1, seq_len, seq_len], data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::{
        ClipTextAttentionWeights, ClipTextLayerWeights, ClipTextWeightsManifest,
    };
    use std::fs;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn test_tokenizer() -> Result<StableDiffusionTokenizer> {
        let path = std::env::temp_dir().join(format!(
            "puppygrad-sd-tokenizer-vocab-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"{
                "<unk>": 0,
                "<|startoftext|>": 1,
                "<|endoftext|>": 2,
                "hello": 3,
                "world": 4,
                "again": 5
            }"#,
        )
        .unwrap();
        let model = WordLevel::from_file(path.to_str().unwrap(), "<unk>".to_string())
            .map_err(|err| StableDiffusionError::Asset(err.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        StableDiffusionTokenizer::from_tokenizer(tokenizer)
    }

    #[test]
    fn loads_clip_bpe_tokenizer_from_vocab_and_merges_without_tokenizer_json() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "puppygrad-sd-bpe-tokenizer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let vocab = root.join("vocab.json");
        let merges = root.join("merges.txt");
        fs::write(
            &vocab,
            r#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1
            }"#,
        )
        .unwrap();
        fs::write(&merges, "#version: 0.2\n").unwrap();

        let tokenizer = StableDiffusionTokenizer::from_diffusers_files(
            root.join("tokenizer.json"),
            &vocab,
            &merges,
        )?;
        let encoded = tokenizer.encode_prompt("")?;

        fs::remove_dir_all(&root).ok();
        assert_eq!(encoded.token_ids[0], 0);
        assert_eq!(encoded.token_ids[1], 1);
        assert_eq!(encoded.token_ids[2], 1);
        Ok(())
    }

    fn tiny_clip_config() -> ClipTextConfig {
        ClipTextConfig {
            class_name: "CLIPTextModel".to_string(),
            vocab_size: 4,
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            max_position_embeddings: 4,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
        }
    }

    fn tiny_clip_weights() -> ClipTextWeights {
        let zeros2x2 = vec![0.0; 4];
        let zeros2 = vec![0.0; 2];
        ClipTextWeights {
            manifest: ClipTextWeightsManifest {
                tensor_count: 20,
                layers: 1,
            },
            token_embedding: vec![0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.25, -0.25],
            position_embedding: vec![0.0; 8],
            layers: vec![ClipTextLayerWeights {
                self_attn: ClipTextAttentionWeights {
                    q_proj_weight: zeros2x2.clone(),
                    q_proj_bias: zeros2.clone(),
                    k_proj_weight: zeros2x2.clone(),
                    k_proj_bias: zeros2.clone(),
                    v_proj_weight: zeros2x2.clone(),
                    v_proj_bias: zeros2.clone(),
                    out_proj_weight: zeros2x2.clone(),
                    out_proj_bias: zeros2.clone(),
                },
                layer_norm1_weight: vec![1.0, 1.0],
                layer_norm1_bias: zeros2.clone(),
                mlp_fc1_weight: zeros2x2.clone(),
                mlp_fc1_bias: zeros2.clone(),
                mlp_fc2_weight: zeros2x2,
                mlp_fc2_bias: zeros2.clone(),
                layer_norm2_weight: vec![1.0, 1.0],
                layer_norm2_bias: zeros2,
            }],
            final_layer_norm_weight: vec![1.0, 1.0],
            final_layer_norm_bias: vec![0.0, 0.0],
        }
    }

    #[test]
    fn tokenizes_prompt_with_clip_special_tokens_and_padding() -> Result<()> {
        let tokenizer = test_tokenizer()?;

        let tokens = tokenizer.encode_prompt_with_max_len("hello world", 6)?;

        assert_eq!(tokens.token_ids, vec![1, 3, 4, 2, 2, 2]);
        assert_eq!(tokens.attention_mask, vec![1, 1, 1, 1, 0, 0]);
        assert!(!tokens.truncated);
        Ok(())
    }

    #[test]
    fn truncates_content_but_preserves_eos() -> Result<()> {
        let tokenizer = test_tokenizer()?;

        let tokens = tokenizer.encode_prompt_with_max_len("hello world again hello", 5)?;

        assert_eq!(tokens.token_ids, vec![1, 3, 4, 5, 2]);
        assert_eq!(tokens.attention_mask, vec![1, 1, 1, 1, 1]);
        assert!(tokens.truncated);
        Ok(())
    }

    #[test]
    fn encodes_negative_prompt_with_same_shape() -> Result<()> {
        let tokenizer = test_tokenizer()?;

        let tokens = tokenizer.encode_conditioning("hello", "")?;

        assert_eq!(tokens.prompt.token_ids.len(), SD1_CLIP_MAX_TOKENS);
        assert_eq!(tokens.negative_prompt.token_ids.len(), SD1_CLIP_MAX_TOKENS);
        validate_clip_token_ids(&tokens.prompt.token_ids)?;
        validate_clip_token_ids(&tokens.negative_prompt.token_ids)?;
        Ok(())
    }

    #[test]
    fn clip_text_encoder_runs_transformer_layers_deterministically() -> Result<()> {
        let encoder = ClipTextEncoder::new(tiny_clip_config(), tiny_clip_weights())?;

        let encoded = encoder.encode_token_ids(&[1, 2])?;

        assert_eq!(encoded.shape(), &[1, 2, 2]);
        assert!(encoded.is_finite());
        let expected = [0.999_995, -0.999_995, -0.999_995, 0.999_995];
        for (actual, expected) in encoded.data().iter().zip(expected.iter()) {
            assert!((*actual - *expected).abs() < 1e-4);
        }
        Ok(())
    }
}
