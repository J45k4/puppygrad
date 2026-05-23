use crate::models::generation::{LogitsSampler, TextGenerationConfig};

use super::{
    BarkCausalTransformer, BarkError, BarkGenerationConfig, BarkHistoryPrompt, BarkRuntimeOptions,
    Result,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkCoarseInput {
    pub semantic_tokens: Vec<usize>,
    pub model_input_ids: Vec<usize>,
}

pub fn build_coarse_input(
    semantic_tokens: &[usize],
    generation_config: &BarkGenerationConfig,
) -> Result<BarkCoarseInput> {
    let semantic = &generation_config.semantic_config;
    let coarse = &generation_config.coarse_acoustics_config;
    if semantic_tokens
        .iter()
        .any(|token| *token >= semantic.semantic_vocab_size)
    {
        return Err(BarkError::InvalidInput(
            "semantic token is outside semantic vocabulary".to_string(),
        ));
    }
    let semantic_tokens = semantic_tokens.to_vec();

    let mut model_input_ids = Vec::with_capacity(coarse.max_coarse_input_length + 1);
    model_input_ids.extend(
        semantic_tokens
            .iter()
            .copied()
            .take(coarse.max_coarse_input_length),
    );
    model_input_ids.resize(
        coarse.max_coarse_input_length,
        coarse.coarse_semantic_pad_token,
    );
    model_input_ids.push(coarse.coarse_infer_token);

    Ok(BarkCoarseInput {
        semantic_tokens,
        model_input_ids,
    })
}

pub fn mask_coarse_logits_for_codebook(
    logits: &mut [f32],
    codebook_index: usize,
    codebook_size: usize,
    semantic_vocab_size: usize,
) -> Result<()> {
    if codebook_size == 0 {
        return Err(BarkError::InvalidInput(
            "codebook_size must be > 0".to_string(),
        ));
    }
    let start = semantic_vocab_size
        + codebook_index
            .checked_mul(codebook_size)
            .ok_or_else(|| BarkError::InvalidInput("coarse codebook range overflow".to_string()))?;
    let end = start + codebook_size;
    if end > logits.len() {
        return Err(BarkError::InvalidInput(format!(
            "coarse codebook range {start}..{end} exceeds logits length {}",
            logits.len()
        )));
    }
    for (idx, logit) in logits.iter_mut().enumerate() {
        if idx < start || idx >= end {
            *logit = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

pub fn deinterleave_coarse_codes(
    generated_tokens: &[usize],
    n_codebooks: usize,
    codebook_size: usize,
    semantic_vocab_size: usize,
) -> Result<Vec<Vec<usize>>> {
    if n_codebooks == 0 || codebook_size == 0 {
        return Err(BarkError::InvalidInput(
            "coarse codebook count and size must be > 0".to_string(),
        ));
    }
    if !generated_tokens.len().is_multiple_of(n_codebooks) {
        return Err(BarkError::InvalidInput(format!(
            "{} generated coarse tokens are not divisible by {n_codebooks} codebooks",
            generated_tokens.len()
        )));
    }
    let frames = generated_tokens.len() / n_codebooks;
    let mut codes = vec![vec![0; frames]; n_codebooks];
    for frame in 0..frames {
        for codebook in 0..n_codebooks {
            let token = generated_tokens[frame * n_codebooks + codebook];
            let offset = semantic_vocab_size + codebook * codebook_size;
            if token < offset || token >= offset + codebook_size {
                return Err(BarkError::InvalidInput(format!(
                    "coarse token {token} is outside codebook {codebook} range"
                )));
            }
            codes[codebook][frame] = token - offset;
        }
    }
    Ok(codes)
}

pub fn generate_coarse_codes(
    model: &BarkCausalTransformer,
    input: &BarkCoarseInput,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
) -> Result<Vec<Vec<usize>>> {
    generate_coarse_codes_with_history(model, input, generation_config, options, None)
}

pub fn generate_coarse_codes_with_history(
    model: &BarkCausalTransformer,
    input: &BarkCoarseInput,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
    history_prompt: Option<&BarkHistoryPrompt>,
) -> Result<Vec<Vec<usize>>> {
    generate_coarse_codes_with_history_and_progress(
        model,
        input,
        generation_config,
        options,
        history_prompt,
        |_, _| {},
    )
}

pub fn generate_coarse_codes_with_history_and_progress(
    model: &BarkCausalTransformer,
    input: &BarkCoarseInput,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
    history_prompt: Option<&BarkHistoryPrompt>,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<Vec<usize>>> {
    let semantic = &generation_config.semantic_config;
    let coarse = &generation_config.coarse_acoustics_config;
    let n_codebooks = coarse.n_coarse_codebooks;
    let semantic_to_coarse_ratio = coarse.semantic_to_coarse_token_ratio(semantic.semantic_rate_hz);
    let max_new_tokens = coarse_output_token_count(
        input.semantic_tokens.len(),
        semantic_to_coarse_ratio,
        n_codebooks,
    )?;
    if max_new_tokens == 0 {
        return Err(BarkError::InvalidInput(
            "coarse generation requires at least one semantic token".to_string(),
        ));
    }
    let mut sampler = LogitsSampler::new(options.seed);
    let sampler_config = TextGenerationConfig {
        max_new_tokens,
        eos_token_id: None,
        temperature: options.coarse_sampling.temperature,
        top_p: options.coarse_sampling.top_p,
        top_k: options.coarse_sampling.top_k,
        seed: options.seed,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
    };
    sampler_config
        .validate()
        .map_err(|err| BarkError::InvalidInput(err.to_string()))?;

    let max_semantic_history =
        (coarse.max_coarse_history as f32 / semantic_to_coarse_ratio).floor() as usize;
    let mut semantic_output = history_prompt
        .map(|prompt| prompt.semantic_prompt.clone())
        .unwrap_or_default();
    semantic_output.extend_from_slice(&input.semantic_tokens);
    let semantic_output = semantic_output
        .iter()
        .map(|token| {
            if *token == semantic.semantic_pad_token {
                coarse.coarse_semantic_pad_token
            } else {
                *token
            }
        })
        .collect::<Vec<_>>();
    let mut history_coarse_tokens = history_prompt.map_or_else(Vec::new, |prompt| {
        flatten_coarse_history(
            prompt,
            generation_config.codebook_size,
            semantic.semantic_vocab_size,
        )
    });
    let semantic_history_len = history_prompt.map_or(0, |prompt| prompt.semantic_prompt.len());
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut cache = model.new_kv_cache();
    while generated.len() < max_new_tokens {
        let remaining = max_new_tokens - generated.len();
        let window_tokens = coarse.sliding_window_len.min(remaining);
        let semantic_idx = semantic_history_len
            + (generated.len() as f32 / semantic_to_coarse_ratio).round() as usize;
        let mut context = build_coarse_window_input(
            &semantic_output,
            &history_coarse_tokens,
            semantic_idx,
            max_semantic_history,
            generation_config,
        );
        truncate_coarse_context(
            &mut context,
            coarse.max_coarse_input_length + 1 + coarse.max_coarse_history,
            model.config.block_size,
        );
        cache.clear();
        let context_embeddings = super::embedding_lookup(
            &context,
            &model.weights.token_embedding,
            model.config.input_vocab_size,
            model.config.hidden_size,
        )?;
        let mut logits = model.prefill_cached_last_logits_from_embeddings(
            &context_embeddings,
            context.len(),
            &mut cache,
        )?;
        for _ in 0..window_tokens {
            let codebook = next_coarse_codebook(generated.len(), n_codebooks);
            mask_coarse_logits_for_codebook(
                &mut logits,
                codebook,
                generation_config.codebook_size,
                semantic.semantic_vocab_size,
            )?;
            let next = sampler
                .select_next_token(&logits, &generated, &sampler_config)
                .map_err(|err| BarkError::InvalidInput(err.to_string()))?;
            generated.push(next);
            if should_report_progress(generated.len(), max_new_tokens, 20) {
                progress(generated.len(), max_new_tokens);
            }
            history_coarse_tokens.push(next);
            let cache_valid =
                append_coarse_token_to_window_context(&mut context, next, model.config.block_size);
            if generated.len() < max_new_tokens && cache_valid {
                logits = model.cached_last_logits(&[next], &mut cache)?;
            } else if generated.len() < max_new_tokens {
                cache.clear();
                logits = model.cached_last_logits(&context, &mut cache)?;
            }
        }
    }
    deinterleave_coarse_codes(
        &generated,
        n_codebooks,
        generation_config.codebook_size,
        semantic.semantic_vocab_size,
    )
}

fn next_coarse_codebook(generated_len: usize, n_codebooks: usize) -> usize {
    generated_len % n_codebooks
}

fn should_report_progress(current: usize, total: usize, interval: usize) -> bool {
    current == 1 || current == total || current.is_multiple_of(interval)
}

fn append_coarse_token_to_window_context(
    context: &mut Vec<usize>,
    token: usize,
    block_size: usize,
) -> bool {
    context.push(token);
    if context.len() <= block_size {
        return true;
    }
    truncate_coarse_context(context, block_size, block_size);
    false
}

fn flatten_coarse_history(
    history_prompt: &BarkHistoryPrompt,
    codebook_size: usize,
    semantic_vocab_size: usize,
) -> Vec<usize> {
    let frames = history_prompt
        .coarse_prompt
        .first()
        .map(Vec::len)
        .unwrap_or_default();
    let mut tokens = Vec::with_capacity(history_prompt.coarse_prompt.len() * frames);
    for frame in 0..frames {
        for (codebook, codes) in history_prompt.coarse_prompt.iter().enumerate() {
            tokens.push(semantic_vocab_size + codebook * codebook_size + codes[frame]);
        }
    }
    tokens
}

pub fn coarse_output_token_count(
    semantic_token_count: usize,
    semantic_to_coarse_ratio: f32,
    n_codebooks: usize,
) -> Result<usize> {
    if !semantic_to_coarse_ratio.is_finite() || semantic_to_coarse_ratio <= 0.0 {
        return Err(BarkError::InvalidInput(
            "semantic-to-coarse ratio must be finite and > 0".to_string(),
        ));
    }
    if n_codebooks == 0 {
        return Err(BarkError::InvalidInput(
            "coarse codebook count must be > 0".to_string(),
        ));
    }
    let frames = (semantic_token_count as f32 * semantic_to_coarse_ratio / n_codebooks as f32)
        .floor() as usize;
    Ok(frames * n_codebooks)
}

pub fn build_coarse_window_input(
    semantic_output: &[usize],
    generated_coarse_tokens: &[usize],
    semantic_idx: usize,
    max_semantic_history: usize,
    generation_config: &BarkGenerationConfig,
) -> Vec<usize> {
    let coarse = &generation_config.coarse_acoustics_config;
    let start = semantic_idx.saturating_sub(max_semantic_history);
    let mut input = semantic_output
        .get(start..)
        .unwrap_or_default()
        .iter()
        .copied()
        .take(coarse.max_coarse_input_length)
        .collect::<Vec<_>>();
    input.resize(
        coarse.max_coarse_input_length,
        coarse.coarse_semantic_pad_token,
    );
    input.push(coarse.coarse_infer_token);
    let history_start = generated_coarse_tokens
        .len()
        .saturating_sub(coarse.max_coarse_history);
    input.extend_from_slice(&generated_coarse_tokens[history_start..]);
    input
}

pub fn truncate_coarse_context(
    context: &mut Vec<usize>,
    configured_limit: usize,
    block_size: usize,
) {
    let limit = configured_limit.min(block_size).max(1);
    if context.len() > limit {
        let drain = context.len() - limit;
        context.drain(..drain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bark::{
        BarkCoarseGenerationConfig, BarkFineGenerationConfig, BarkSemanticGenerationConfig,
    };

    #[test]
    fn coarse_input_pads_and_appends_infer_token() -> Result<()> {
        let config = generation_config();

        let input = build_coarse_input(&[1, 2], &config)?;

        assert_eq!(input.semantic_tokens, vec![1, 2]);
        assert_eq!(input.model_input_ids, vec![1, 2, 99, 99, 77]);
        Ok(())
    }

    #[test]
    fn coarse_input_keeps_full_semantic_sequence_for_output_length() -> Result<()> {
        let config = generation_config();

        let input = build_coarse_input(&[1, 2, 3, 4, 5, 6], &config)?;

        assert_eq!(input.semantic_tokens, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(input.model_input_ids, vec![1, 2, 3, 4, 77]);
        Ok(())
    }

    #[test]
    fn coarse_mask_keeps_only_active_codebook_range() -> Result<()> {
        let mut logits = vec![0.0; 12];

        mask_coarse_logits_for_codebook(&mut logits, 1, 4, 2)?;

        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert_eq!(logits[6], 0.0);
        assert_eq!(logits[9], 0.0);
        assert_eq!(logits[10], f32::NEG_INFINITY);
        Ok(())
    }

    #[test]
    fn deinterleaves_coarse_tokens_into_codebook_rows() -> Result<()> {
        let codes = deinterleave_coarse_codes(&[10, 14, 11, 15], 2, 4, 10)?;

        assert_eq!(codes, vec![vec![0, 1], vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn coarse_history_flattens_to_interleaved_token_ids() {
        let history = BarkHistoryPrompt {
            semantic_prompt: vec![1, 2],
            coarse_prompt: vec![vec![1, 2], vec![3, 0]],
            fine_prompt: vec![vec![0], vec![0], vec![0], vec![0]],
        };

        let tokens = flatten_coarse_history(&history, 4, 10);

        assert_eq!(tokens, vec![11, 17, 12, 14]);
    }

    #[test]
    fn coarse_context_truncates_to_configured_and_block_limits() {
        let mut context = vec![1, 2, 3, 4, 5];

        truncate_coarse_context(&mut context, 4, 3);

        assert_eq!(context, vec![3, 4, 5]);
    }

    #[test]
    fn coarse_output_token_count_matches_transformers_rounding() -> Result<()> {
        let config = generation_config();
        let ratio = config
            .coarse_acoustics_config
            .semantic_to_coarse_token_ratio(config.semantic_config.semantic_rate_hz);

        let tokens = coarse_output_token_count(1, ratio, 2)?;

        assert_eq!(tokens, 2);
        Ok(())
    }

    #[test]
    fn coarse_window_input_matches_transformers_no_history_layout() {
        let config = generation_config();

        let input = build_coarse_window_input(&[1, 2, 3, 4, 5], &[10, 14, 11], 3, 2, &config);

        assert_eq!(input, vec![2, 3, 4, 5, 77, 10, 14, 11]);
    }

    #[test]
    fn coarse_codebook_selection_uses_generated_position_not_context_length() {
        let codebooks = (0..8)
            .map(|generated_len| next_coarse_codebook(generated_len, 2))
            .collect::<Vec<_>>();

        assert_eq!(codebooks, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn coarse_window_context_keeps_cache_valid_past_history_cap_until_block_limit() {
        let mut context = vec![1, 2, 3, 4];

        let cache_valid = append_coarse_token_to_window_context(&mut context, 5, 6);

        assert!(cache_valid);
        assert_eq!(context, vec![1, 2, 3, 4, 5]);

        let cache_valid = append_coarse_token_to_window_context(&mut context, 6, 6);

        assert!(cache_valid);
        assert_eq!(context, vec![1, 2, 3, 4, 5, 6]);

        let cache_valid = append_coarse_token_to_window_context(&mut context, 7, 6);

        assert!(!cache_valid);
        assert_eq!(context, vec![2, 3, 4, 5, 6, 7]);
    }

    fn generation_config() -> BarkGenerationConfig {
        BarkGenerationConfig {
            sample_rate: 24_000,
            codebook_size: 4,
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
                coarse_infer_token: 77,
                coarse_rate_hz: 75,
                coarse_semantic_pad_token: 99,
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
}
