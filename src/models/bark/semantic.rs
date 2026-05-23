use crate::models::generation::{LogitsSampler, TextGenerationConfig};

use super::{
    BarkCausalTransformer, BarkError, BarkGenerationConfig, BarkHistoryPrompt, BarkRuntimeOptions,
    BarkTokenizer, Result,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkSemanticInput {
    pub text_semantic_ids: Vec<usize>,
    pub semantic_history_ids: Vec<usize>,
    pub model_input_ids: Vec<usize>,
}

pub fn build_semantic_input(
    tokenizer: &BarkTokenizer,
    text: &str,
    generation_config: &BarkGenerationConfig,
) -> Result<BarkSemanticInput> {
    build_semantic_input_with_history(tokenizer, text, generation_config, None)
}

pub fn build_semantic_input_with_history(
    tokenizer: &BarkTokenizer,
    text: &str,
    generation_config: &BarkGenerationConfig,
    history_prompt: Option<&BarkHistoryPrompt>,
) -> Result<BarkSemanticInput> {
    let semantic = &generation_config.semantic_config;
    let mut text_semantic_ids = tokenizer.encode_for_semantic_model(
        text,
        semantic.text_encoding_offset,
        semantic.text_pad_token,
    )?;
    text_semantic_ids.truncate(semantic.max_input_semantic_length);

    let mut model_input_ids = Vec::with_capacity(semantic.max_input_semantic_length + 1);
    model_input_ids.extend_from_slice(&text_semantic_ids);
    model_input_ids.resize(semantic.max_input_semantic_length, semantic.text_pad_token);
    model_input_ids.push(semantic.semantic_infer_token);
    let semantic_history_ids = semantic_history_ids(history_prompt, generation_config);

    Ok(BarkSemanticInput {
        text_semantic_ids,
        semantic_history_ids,
        model_input_ids,
    })
}

fn semantic_history_ids(
    history_prompt: Option<&BarkHistoryPrompt>,
    generation_config: &BarkGenerationConfig,
) -> Vec<usize> {
    let semantic = &generation_config.semantic_config;
    let Some(history_prompt) = history_prompt else {
        return vec![semantic.semantic_pad_token; semantic.max_input_semantic_length];
    };
    let start = history_prompt
        .semantic_prompt
        .len()
        .saturating_sub(semantic.max_input_semantic_length);
    let mut ids = history_prompt.semantic_prompt[start..].to_vec();
    ids.resize(
        semantic.max_input_semantic_length,
        semantic.semantic_pad_token,
    );
    ids
}

pub fn build_semantic_input_embeddings(
    model: &BarkCausalTransformer,
    input: &BarkSemanticInput,
    generation_config: &BarkGenerationConfig,
) -> Result<Vec<f32>> {
    let semantic = &generation_config.semantic_config;
    let hidden = model.config.hidden_size;
    if input.model_input_ids.len() != semantic.max_input_semantic_length + 1 {
        return Err(BarkError::InvalidInput(
            "semantic model input length does not match max_input_semantic_length + 1".to_string(),
        ));
    }
    if input.semantic_history_ids.len() != semantic.max_input_semantic_length {
        return Err(BarkError::InvalidInput(
            "semantic history length does not match max_input_semantic_length".to_string(),
        ));
    }
    let mut embeddings = vec![0.0; input.model_input_ids.len() * hidden];
    for pos in 0..semantic.max_input_semantic_length {
        add_token_embedding(
            model,
            input.model_input_ids[pos],
            &mut embeddings[pos * hidden..(pos + 1) * hidden],
        )?;
        add_token_embedding(
            model,
            input.semantic_history_ids[pos],
            &mut embeddings[pos * hidden..(pos + 1) * hidden],
        )?;
    }
    add_token_embedding(
        model,
        semantic.semantic_infer_token,
        &mut embeddings[semantic.max_input_semantic_length * hidden
            ..(semantic.max_input_semantic_length + 1) * hidden],
    )?;
    Ok(embeddings)
}

pub fn generate_semantic_tokens(
    model: &BarkCausalTransformer,
    input: &BarkSemanticInput,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
) -> Result<Vec<usize>> {
    generate_semantic_tokens_with_progress(model, input, generation_config, options, |_, _| {})
}

pub fn generate_semantic_tokens_with_progress(
    model: &BarkCausalTransformer,
    input: &BarkSemanticInput,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<usize>> {
    let semantic = &generation_config.semantic_config;
    let max_new_tokens = options
        .max_semantic_tokens
        .unwrap_or(semantic.max_new_tokens)
        .min(semantic.max_new_tokens);
    if max_new_tokens == 0 {
        return Err(BarkError::InvalidInput(
            "semantic max_new_tokens must be > 0".to_string(),
        ));
    }
    let mut sampler = LogitsSampler::new(options.seed);
    let sampler_config = TextGenerationConfig {
        max_new_tokens,
        eos_token_id: Some(semantic.eos_token_id),
        temperature: options.semantic_sampling.temperature,
        top_p: options.semantic_sampling.top_p,
        top_k: options.semantic_sampling.top_k,
        seed: options.seed,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
    };
    sampler_config
        .validate()
        .map_err(|err| BarkError::InvalidInput(err.to_string()))?;

    let context_embeddings = build_semantic_input_embeddings(model, input, generation_config)?;
    let mut cache = model.new_kv_cache();
    let mut logits = model.prefill_cached_last_logits_from_embeddings(
        &context_embeddings,
        input.model_input_ids.len(),
        &mut cache,
    )?;
    let mut generated = Vec::new();
    let hidden = model.config.hidden_size;
    let mut next_embedding = vec![0.0; hidden];
    for _ in 0..max_new_tokens {
        mask_semantic_logits(&mut logits, semantic.semantic_vocab_size);
        let next = sampler
            .select_next_token(&logits, &generated, &sampler_config)
            .map_err(|err| BarkError::InvalidInput(err.to_string()))?;
        if next == semantic.eos_token_id {
            break;
        }
        generated.push(next);
        if should_report_progress(generated.len(), max_new_tokens, 16) {
            progress(generated.len(), max_new_tokens);
        }
        if generated.len() == max_new_tokens {
            break;
        }
        next_embedding.fill(0.0);
        add_token_embedding(model, next, &mut next_embedding)?;
        logits = model.cached_last_logits_from_embeddings(&next_embedding, 1, &mut cache)?;
    }
    Ok(generated)
}

fn should_report_progress(current: usize, total: usize, interval: usize) -> bool {
    current == 1 || current == total || current.is_multiple_of(interval)
}

pub fn mask_semantic_logits(logits: &mut [f32], semantic_vocab_size: usize) {
    for logit in logits.iter_mut().skip(semantic_vocab_size + 1) {
        *logit = f32::NEG_INFINITY;
    }
}

fn add_token_embedding(
    model: &BarkCausalTransformer,
    token_id: usize,
    dst: &mut [f32],
) -> Result<()> {
    let hidden = model.config.hidden_size;
    if dst.len() != hidden {
        return Err(BarkError::InvalidInput(
            "semantic embedding destination shape mismatch".to_string(),
        ));
    }
    if token_id >= model.config.input_vocab_size {
        return Err(BarkError::InvalidInput(format!(
            "semantic token id {token_id} exceeds transformer input vocab size {}",
            model.config.input_vocab_size
        )));
    }
    let start = token_id * hidden;
    for dim in 0..hidden {
        dst[dim] += model.weights.token_embedding[start + dim];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bark::{
        BarkCausalTransformer, BarkCausalTransformerLayerWeights, BarkCausalTransformerWeights,
        BarkCoarseGenerationConfig, BarkFineGenerationConfig, BarkSemanticGenerationConfig,
        BarkSubModelConfig,
    };
    use std::collections::HashMap;

    #[test]
    fn semantic_input_offsets_pads_and_appends_infer_token() -> Result<()> {
        let tokenizer = tokenizer();
        let config = generation_config();

        let input = build_semantic_input(&tokenizer, "hello", &config)?;

        assert_eq!(input.text_semantic_ids, vec![104, 999, 999, 999]);
        assert_eq!(input.semantic_history_ids, vec![10, 10, 10, 10]);
        assert_eq!(input.model_input_ids, vec![104, 999, 999, 999, 555]);
        Ok(())
    }

    #[test]
    fn semantic_input_uses_tail_history_prompt_and_pads() -> Result<()> {
        let tokenizer = tokenizer();
        let config = generation_config();
        let history = super::super::BarkHistoryPrompt {
            semantic_prompt: vec![1, 2, 3],
            coarse_prompt: vec![vec![0], vec![0]],
            fine_prompt: vec![
                vec![0],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
            ],
        };

        let input =
            build_semantic_input_with_history(&tokenizer, "hello", &config, Some(&history))?;

        assert_eq!(input.semantic_history_ids, vec![1, 2, 3, 10]);
        Ok(())
    }

    #[test]
    fn semantic_embeddings_sum_text_and_history_then_append_infer() -> Result<()> {
        let tokenizer = tokenizer();
        let config = generation_config();
        let input = build_semantic_input(&tokenizer, "hello", &config)?;
        let transformer_config = BarkSubModelConfig {
            block_size: 16,
            input_vocab_size: 1000,
            output_vocab_size: 12,
            num_layers: 1,
            num_heads: 1,
            hidden_size: 2,
            dropout: 0.0,
            bias: true,
            use_cache: false,
            model_type: Some("semantic".to_string()),
        };
        let model = BarkCausalTransformer::new(
            transformer_config.clone(),
            tiny_weights(&transformer_config),
        )?;

        let embeddings = build_semantic_input_embeddings(&model, &input, &config)?;

        assert_eq!(embeddings.len(), 5 * 2);
        Ok(())
    }

    #[test]
    fn semantic_generation_masks_to_vocab_and_respects_max_tokens() -> Result<()> {
        let tokenizer = tokenizer();
        let config = generation_config();
        let input = build_semantic_input(&tokenizer, "hello", &config)?;
        let transformer_config = BarkSubModelConfig {
            block_size: 16,
            input_vocab_size: 1000,
            output_vocab_size: 12,
            num_layers: 1,
            num_heads: 1,
            hidden_size: 2,
            dropout: 0.0,
            bias: true,
            use_cache: false,
            model_type: Some("semantic".to_string()),
        };
        let model = BarkCausalTransformer::new(
            transformer_config.clone(),
            tiny_weights(&transformer_config),
        )?;
        let mut options = BarkRuntimeOptions::from_generation_config("hello", None, 1, &config);
        options.max_semantic_tokens = Some(2);
        options.semantic_sampling.temperature = 0.0;

        let generated = generate_semantic_tokens(&model, &input, &config, &options)?;

        assert!(generated.len() <= 2);
        assert!(generated.iter().all(|token| *token < 10));
        Ok(())
    }

    #[test]
    fn semantic_generation_does_not_decode_past_block_on_final_token() -> Result<()> {
        let tokenizer = tokenizer();
        let mut config = generation_config();
        config.semantic_config.max_new_tokens = 11;
        let input = build_semantic_input(&tokenizer, "hello", &config)?;
        let transformer_config = BarkSubModelConfig {
            block_size: input.model_input_ids.len() + 11,
            input_vocab_size: 1000,
            output_vocab_size: 12,
            num_layers: 1,
            num_heads: 1,
            hidden_size: 2,
            dropout: 0.0,
            bias: true,
            use_cache: false,
            model_type: Some("semantic".to_string()),
        };
        let model = BarkCausalTransformer::new(
            transformer_config.clone(),
            tiny_weights(&transformer_config),
        )?;
        let mut options = BarkRuntimeOptions::from_generation_config("hello", None, 1, &config);
        options.max_semantic_tokens = Some(11);
        options.semantic_sampling.temperature = 0.0;

        let generated = generate_semantic_tokens(&model, &input, &config, &options)?;

        assert_eq!(generated.len(), 11);
        Ok(())
    }

    #[test]
    fn semantic_logit_mask_invalidates_tail() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];

        mask_semantic_logits(&mut logits, 2);

        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    fn tokenizer() -> BarkTokenizer {
        let mut vocab = HashMap::new();
        for (idx, token) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"]
            .into_iter()
            .enumerate()
        {
            vocab.insert(token.to_string(), idx);
        }
        BarkTokenizer::from_vocab(vocab, 8).unwrap()
    }

    fn generation_config() -> BarkGenerationConfig {
        BarkGenerationConfig {
            sample_rate: 24_000,
            codebook_size: 1024,
            semantic_config: BarkSemanticGenerationConfig {
                eos_token_id: 10,
                max_input_semantic_length: 4,
                max_new_tokens: 8,
                semantic_infer_token: 555,
                semantic_pad_token: 10,
                semantic_rate_hz: 49.9,
                semantic_vocab_size: 10,
                text_encoding_offset: 100,
                text_pad_token: 999,
                temperature: 0.7,
                top_k: 50,
                top_p: 1.0,
            },
            coarse_acoustics_config: BarkCoarseGenerationConfig {
                coarse_infer_token: 20,
                coarse_rate_hz: 75,
                coarse_semantic_pad_token: 30,
                max_coarse_history: 4,
                max_coarse_input_length: 4,
                n_coarse_codebooks: 2,
                sliding_window_len: 2,
                temperature: 0.7,
                top_k: 50,
                top_p: 1.0,
            },
            fine_acoustics_config: BarkFineGenerationConfig {
                max_fine_history_length: 4,
                max_fine_input_length: 4,
                n_fine_codebooks: 8,
                temperature: 0.5,
                top_k: 50,
                top_p: 1.0,
            },
            model_type: Some("bark".to_string()),
        }
    }

    fn tiny_weights(config: &BarkSubModelConfig) -> BarkCausalTransformerWeights {
        let hidden = config.hidden_size;
        let mlp_hidden = hidden * 4;
        BarkCausalTransformerWeights {
            token_embedding: vec![0.01; config.input_vocab_size * hidden],
            positional_embedding: vec![0.0; config.block_size * hidden],
            layers: vec![BarkCausalTransformerLayerWeights {
                attention_layer_norm_weight: vec![1.0; hidden],
                attention_layer_norm_bias: Some(vec![0.0; hidden]),
                attention_qkv_weight: vec![0.0; hidden * 3 * hidden],
                attention_qkv_bias: Some(vec![0.0; hidden * 3]),
                attention_output_weight: vec![0.0; hidden * hidden],
                attention_output_bias: Some(vec![0.0; hidden]),
                mlp_layer_norm_weight: vec![1.0; hidden],
                mlp_layer_norm_bias: Some(vec![0.0; hidden]),
                mlp_in_weight: vec![0.0; mlp_hidden * hidden],
                mlp_in_bias: Some(vec![0.0; mlp_hidden]),
                mlp_out_weight: vec![0.0; hidden * mlp_hidden],
                mlp_out_bias: Some(vec![0.0; hidden]),
            }],
            final_layer_norm_weight: vec![1.0; hidden],
            final_layer_norm_bias: Some(vec![0.0; hidden]),
            lm_head_weight: (0..config.output_vocab_size)
                .flat_map(|token| [token as f32, 0.0])
                .collect(),
        }
    }
}
