use crate::models::generation::{argmax_logits, LogitsSampler, TextGenerationConfig};

use super::{
    BarkCausalTransformer, BarkCausalTransformerWeights, BarkError, BarkFineGenerationConfig,
    BarkFineSubModelConfig, BarkFineTransformerWeights, BarkGenerationConfig, BarkHistoryPrompt,
    BarkRuntimeOptions, Result,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkFineCodeMatrix {
    pub codebooks: Vec<Vec<usize>>,
}

impl BarkFineCodeMatrix {
    pub fn frames(&self) -> usize {
        self.codebooks.first().map(Vec::len).unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkFineTransformer {
    pub config: BarkFineSubModelConfig,
    hidden_model: BarkCausalTransformer,
    input_embeddings: Vec<Vec<f32>>,
    lm_head_weights: Vec<Vec<f32>>,
}

impl BarkFineTransformer {
    pub fn new(
        config: BarkFineSubModelConfig,
        weights: BarkFineTransformerWeights,
    ) -> Result<Self> {
        Self::new_with_threads(config, weights, 1)
    }

    pub fn new_with_threads(
        config: BarkFineSubModelConfig,
        weights: BarkFineTransformerWeights,
        threads: usize,
    ) -> Result<Self> {
        let base = config.base.clone();
        if weights.input_embeddings.len() != config.n_codes_total {
            return Err(BarkError::InvalidInput(format!(
                "fine input embedding count {} does not match n_codes_total {}",
                weights.input_embeddings.len(),
                config.n_codes_total
            )));
        }
        if weights.lm_head_weights.len() != config.n_codes_total - config.n_codes_given {
            return Err(BarkError::InvalidInput(format!(
                "fine lm head count {} does not match generated codebook count {}",
                weights.lm_head_weights.len(),
                config.n_codes_total - config.n_codes_given
            )));
        }
        let hidden_model = BarkCausalTransformer::new_bidirectional_with_threads(
            base.clone(),
            BarkCausalTransformerWeights {
                token_embedding: vec![0.0; base.input_vocab_size * base.hidden_size],
                positional_embedding: weights.positional_embedding,
                layers: weights.layers,
                final_layer_norm_weight: weights.final_layer_norm_weight,
                final_layer_norm_bias: Some(weights.final_layer_norm_bias),
                lm_head_weight: weights.lm_head_weights[0].clone(),
            },
            threads,
        )?;
        Ok(Self {
            config,
            hidden_model,
            input_embeddings: weights.input_embeddings,
            lm_head_weights: weights.lm_head_weights,
        })
    }

    pub fn forward_logits_for_codebook(
        &self,
        codebook_idx: usize,
        input_frames: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        if codebook_idx < self.config.n_codes_given || codebook_idx >= self.config.n_codes_total {
            return Err(BarkError::InvalidInput(format!(
                "fine codebook_idx {codebook_idx} must be in {}..{}",
                self.config.n_codes_given, self.config.n_codes_total
            )));
        }
        if input_frames.is_empty() {
            return Err(BarkError::InvalidInput(
                "fine input frames must not be empty".to_string(),
            ));
        }
        let base = &self.config.base;
        if input_frames.len() > base.block_size {
            return Err(BarkError::InvalidInput(format!(
                "fine input length {} exceeds block_size {}",
                input_frames.len(),
                base.block_size
            )));
        }
        let hidden = base.hidden_size;
        let mut embeddings = vec![0.0; input_frames.len() * hidden];
        for (frame, codes) in input_frames.iter().enumerate() {
            if codes.len() != self.config.n_codes_total {
                return Err(BarkError::InvalidInput(format!(
                    "fine input frame has {} codebooks, expected {}",
                    codes.len(),
                    self.config.n_codes_total
                )));
            }
            for (codebook, code) in codes.iter().copied().enumerate().take(codebook_idx + 1) {
                if code >= base.input_vocab_size {
                    return Err(BarkError::InvalidInput(format!(
                        "fine code {code} exceeds input vocab size {}",
                        base.input_vocab_size
                    )));
                }
                let src = &self.input_embeddings[codebook][code * hidden..(code + 1) * hidden];
                let dst = &mut embeddings[frame * hidden..(frame + 1) * hidden];
                for dim in 0..hidden {
                    dst[dim] += src[dim];
                }
            }
        }
        let hidden_states = self
            .hidden_model
            .forward_hidden_from_embeddings_with_attention_mask(
                &embeddings,
                input_frames.len(),
                0,
                None,
            )?;
        let head = codebook_idx - self.config.n_codes_given;
        self.hidden_model.logits_from_hidden_with_head(
            &hidden_states,
            input_frames.len(),
            &self.lm_head_weights[head],
        )
    }
}

pub fn build_fine_code_matrix(
    coarse_codes: &[Vec<usize>],
    config: &BarkFineGenerationConfig,
) -> Result<BarkFineCodeMatrix> {
    if coarse_codes.is_empty() {
        return Err(BarkError::InvalidInput(
            "coarse codes must contain at least one codebook".to_string(),
        ));
    }
    if coarse_codes.len() > config.n_fine_codebooks {
        return Err(BarkError::InvalidInput(format!(
            "{} coarse codebooks exceeds {} fine codebooks",
            coarse_codes.len(),
            config.n_fine_codebooks
        )));
    }
    let frames = coarse_codes[0].len();
    if frames == 0 {
        return Err(BarkError::InvalidInput(
            "coarse codes must contain at least one frame".to_string(),
        ));
    }
    if coarse_codes.iter().any(|codes| codes.len() != frames) {
        return Err(BarkError::InvalidInput(
            "coarse codebooks must have equal frame counts".to_string(),
        ));
    }
    let mut codebooks = coarse_codes.to_vec();
    codebooks.resize_with(config.n_fine_codebooks, || vec![0; frames]);
    Ok(BarkFineCodeMatrix { codebooks })
}

pub fn generate_fine_codes(
    model: &BarkFineTransformer,
    coarse_codes: &[Vec<usize>],
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
) -> Result<BarkFineCodeMatrix> {
    generate_fine_codes_with_history(model, coarse_codes, generation_config, options, None)
}

pub fn generate_fine_codes_with_history(
    model: &BarkFineTransformer,
    coarse_codes: &[Vec<usize>],
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
    history_prompt: Option<&BarkHistoryPrompt>,
) -> Result<BarkFineCodeMatrix> {
    generate_fine_codes_with_history_and_progress(
        model,
        coarse_codes,
        generation_config,
        options,
        history_prompt,
        |_, _| {},
    )
}

pub fn generate_fine_codes_with_history_and_progress(
    model: &BarkFineTransformer,
    coarse_codes: &[Vec<usize>],
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
    history_prompt: Option<&BarkHistoryPrompt>,
    mut progress: impl FnMut(usize, usize),
) -> Result<BarkFineCodeMatrix> {
    let fine = &generation_config.fine_acoustics_config;
    let mut matrix = build_fine_input_matrix(coarse_codes, fine, generation_config.codebook_size)?;
    let given = coarse_codes.len();
    let history_frames = prepend_fine_history(&mut matrix, history_prompt, fine)?;
    let mut sampler = LogitsSampler::new(options.seed);
    let sampler_config =
        fine_sampler_config(matrix.frames() * (fine.n_fine_codebooks - given), options);
    sampler_config
        .validate()
        .map_err(|err| BarkError::InvalidInput(err.to_string()))?;

    let original_frames = matrix.frames();
    let mut padded_frames = original_frames;
    let mut remove_from_end = 0usize;
    if padded_frames < fine.max_fine_input_length {
        remove_from_end = fine.max_fine_input_length - padded_frames;
        for row in &mut matrix.codebooks {
            row.resize(fine.max_fine_input_length, generation_config.codebook_size);
        }
        padded_frames = fine.max_fine_input_length;
    }
    let loops = fine_generation_loop_count(
        original_frames,
        fine.max_fine_history_length,
        fine.max_fine_input_length,
    );
    let mut history = Vec::new();
    let total_passes = loops * fine.n_fine_codebooks.saturating_sub(given);
    let mut completed_passes = 0usize;
    progress(completed_passes, total_passes);
    for outer in 0..loops {
        let start_idx = (outer * fine.max_fine_history_length)
            .min(padded_frames.saturating_sub(fine.max_fine_input_length));
        let start_fill_idx = (outer * fine.max_fine_history_length)
            .min(padded_frames.saturating_sub(fine.max_fine_history_length));
        let rel_start_fill_idx = (start_fill_idx - start_idx).max(
            history_frames
                .saturating_sub(start_idx)
                .min(fine.max_fine_input_length),
        );
        let mut input_buffer = fine_input_frames(
            &matrix,
            start_idx,
            fine.max_fine_input_length,
            generation_config.codebook_size,
        )?;
        for codebook in given..fine.n_fine_codebooks {
            let logits = model.forward_logits_for_codebook(codebook, &input_buffer)?;
            for frame in rel_start_fill_idx..input_buffer.len() {
                let row = &logits[frame * model.config.base.output_vocab_size
                    ..(frame + 1) * model.config.base.output_vocab_size];
                let code = select_fine_code(
                    row,
                    generation_config.codebook_size,
                    &mut sampler,
                    &history,
                    &sampler_config,
                )?;
                input_buffer[frame][codebook] = code;
                history.push(code);
            }
            completed_passes += 1;
            progress(completed_passes, total_passes);
        }
        for codebook in given..fine.n_fine_codebooks {
            for frame in rel_start_fill_idx..input_buffer.len() {
                let dst_frame = start_idx + frame;
                if dst_frame < padded_frames {
                    matrix.codebooks[codebook][dst_frame] = input_buffer[frame][codebook];
                }
            }
        }
    }
    if remove_from_end > 0 {
        for row in &mut matrix.codebooks {
            row.truncate(original_frames);
        }
    }
    if history_frames > 0 {
        for row in &mut matrix.codebooks {
            row.drain(..history_frames.min(row.len()));
        }
    }
    Ok(matrix)
}

fn fine_sampler_config(
    max_new_tokens: usize,
    options: &BarkRuntimeOptions,
) -> TextGenerationConfig {
    TextGenerationConfig {
        max_new_tokens,
        eos_token_id: None,
        temperature: options.fine_sampling.temperature,
        // Hugging Face Bark's fine-acoustics generate path only uses
        // temperature from GenerationConfig; top-k/top-p are ignored there.
        top_p: None,
        top_k: None,
        seed: options.seed,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
    }
}

fn prepend_fine_history(
    matrix: &mut BarkFineCodeMatrix,
    history_prompt: Option<&BarkHistoryPrompt>,
    config: &BarkFineGenerationConfig,
) -> Result<usize> {
    let Some(history_prompt) = history_prompt else {
        return Ok(0);
    };
    let frames = history_prompt
        .fine_prompt
        .first()
        .map(Vec::len)
        .unwrap_or_default();
    let keep = frames.min(config.max_fine_history_length);
    let start = frames.saturating_sub(keep);
    for (row, history) in matrix.codebooks.iter_mut().zip(&history_prompt.fine_prompt) {
        let mut merged = history[start..].to_vec();
        merged.extend_from_slice(row);
        *row = merged;
    }
    Ok(keep)
}

fn build_fine_input_matrix(
    coarse_codes: &[Vec<usize>],
    config: &BarkFineGenerationConfig,
    pad_code: usize,
) -> Result<BarkFineCodeMatrix> {
    let mut matrix = build_fine_code_matrix(coarse_codes, config)?;
    for row in matrix.codebooks.iter_mut().skip(coarse_codes.len()) {
        row.fill(pad_code);
    }
    Ok(matrix)
}

pub fn fine_generation_loop_count(
    frames: usize,
    history_length: usize,
    input_length: usize,
) -> usize {
    if history_length == 0 || input_length == 0 {
        return 1;
    }
    let extra = frames as isize - input_length as isize;
    let loops = (extra as f32 / history_length as f32).ceil().max(0.0) as usize;
    loops + 1
}

pub fn fine_input_frames(
    matrix: &BarkFineCodeMatrix,
    start: usize,
    len: usize,
    pad_code: usize,
) -> Result<Vec<Vec<usize>>> {
    if matrix.codebooks.is_empty() {
        return Err(BarkError::InvalidInput(
            "fine matrix must contain at least one codebook".to_string(),
        ));
    }
    let mut frames = Vec::with_capacity(len);
    for frame in start..start + len {
        let mut row = Vec::with_capacity(matrix.codebooks.len());
        for codebook in &matrix.codebooks {
            row.push(codebook.get(frame).copied().unwrap_or(pad_code));
        }
        frames.push(row);
    }
    Ok(frames)
}

fn select_fine_code(
    logits: &[f32],
    codebook_size: usize,
    sampler: &mut LogitsSampler,
    history: &[usize],
    sampler_config: &TextGenerationConfig,
) -> Result<usize> {
    let end = codebook_size.min(logits.len());
    let mut masked = logits.to_vec();
    mask_fine_logits(&mut masked, end);
    if sampler_config.temperature == 1.0 {
        return Ok(argmax_logits(&masked[..end]));
    }
    let next = sampler
        .select_next_token(&masked, history, sampler_config)
        .map_err(|err| BarkError::InvalidInput(err.to_string()))?;
    Ok(next % codebook_size)
}

pub fn flatten_code_matrix_with_offsets(
    matrix: &BarkFineCodeMatrix,
    codebook_size: usize,
    stop_before: Option<(usize, usize)>,
) -> Result<Vec<usize>> {
    if codebook_size == 0 {
        return Err(BarkError::InvalidInput(
            "codebook_size must be > 0".to_string(),
        ));
    }
    let frames = matrix.frames();
    let mut tokens = Vec::with_capacity(matrix.codebooks.len() * frames);
    for codebook in 0..matrix.codebooks.len() {
        for frame in 0..frames {
            if stop_before.is_some_and(|(stop_codebook, stop_frame)| {
                codebook > stop_codebook || (codebook == stop_codebook && frame >= stop_frame)
            }) {
                return Ok(tokens);
            }
            let code = matrix.codebooks[codebook][frame];
            if code >= codebook_size {
                return Err(BarkError::InvalidInput(format!(
                    "fine code {code} exceeds codebook size {codebook_size}"
                )));
            }
            tokens.push(codebook * codebook_size + code);
        }
    }
    Ok(tokens)
}

pub fn mask_fine_logits(logits: &mut [f32], codebook_size: usize) {
    for logit in logits.iter_mut().skip(codebook_size) {
        *logit = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bark::{BarkCoarseGenerationConfig, BarkSemanticGenerationConfig};

    #[test]
    fn fine_matrix_keeps_coarse_rows_and_pads_remaining_codebooks() -> Result<()> {
        let config = BarkFineGenerationConfig {
            max_fine_history_length: 4,
            max_fine_input_length: 4,
            n_fine_codebooks: 4,
            temperature: 0.5,
            top_k: 50,
            top_p: 1.0,
        };

        let matrix = build_fine_code_matrix(&[vec![1, 2], vec![3, 4]], &config)?;

        assert_eq!(matrix.frames(), 2);
        assert_eq!(
            matrix.codebooks,
            vec![vec![1, 2], vec![3, 4], vec![0, 0], vec![0, 0]]
        );
        Ok(())
    }

    #[test]
    fn flatten_code_matrix_offsets_each_codebook() -> Result<()> {
        let matrix = BarkFineCodeMatrix {
            codebooks: vec![vec![1, 2], vec![3, 4]],
        };

        let tokens = flatten_code_matrix_with_offsets(&matrix, 10, None)?;

        assert_eq!(tokens, vec![1, 2, 13, 14]);
        Ok(())
    }

    #[test]
    fn fine_mask_keeps_first_codebook_size_logits() {
        let mut logits = vec![0.0; 5];

        mask_fine_logits(&mut logits, 3);

        assert_eq!(logits[..3], [0.0, 0.0, 0.0]);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn fine_sampler_config_ignores_top_k_and_top_p_like_transformers() {
        let generation_config = generation_config();
        let mut options =
            BarkRuntimeOptions::from_generation_config("hello", None, 7, &generation_config);
        options.fine_sampling.temperature = 0.5;
        options.fine_sampling.top_k = Some(1);
        options.fine_sampling.top_p = Some(0.01);

        let sampler = fine_sampler_config(12, &options);

        assert_eq!(sampler.max_new_tokens, 12);
        assert_eq!(sampler.temperature, 0.5);
        assert_eq!(sampler.top_k, None);
        assert_eq!(sampler.top_p, None);
    }

    #[test]
    fn fine_input_matrix_uses_codebook_size_as_unknown_future_codebooks() -> Result<()> {
        let config = BarkFineGenerationConfig {
            max_fine_history_length: 4,
            max_fine_input_length: 4,
            n_fine_codebooks: 4,
            temperature: 0.5,
            top_k: 50,
            top_p: 1.0,
        };

        let matrix = build_fine_input_matrix(&[vec![1, 2], vec![3, 4]], &config, 10)?;

        assert_eq!(
            matrix.codebooks,
            vec![vec![1, 2], vec![3, 4], vec![10, 10], vec![10, 10]]
        );
        Ok(())
    }

    #[test]
    fn fine_history_is_prepended_to_all_codebooks() -> Result<()> {
        let config = BarkFineGenerationConfig {
            max_fine_history_length: 1,
            max_fine_input_length: 4,
            n_fine_codebooks: 4,
            temperature: 0.5,
            top_k: 50,
            top_p: 1.0,
        };
        let history = BarkHistoryPrompt {
            semantic_prompt: vec![1],
            coarse_prompt: vec![vec![0], vec![0]],
            fine_prompt: vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]],
        };
        let mut matrix = build_fine_input_matrix(&[vec![9], vec![8]], &config, 10)?;

        let frames = prepend_fine_history(&mut matrix, Some(&history), &config)?;

        assert_eq!(frames, 1);
        assert_eq!(
            matrix.codebooks,
            vec![vec![2, 9], vec![4, 8], vec![6, 10], vec![8, 10]]
        );
        Ok(())
    }

    #[test]
    fn fine_input_frames_transpose_codebook_major_matrix_to_frame_rows() -> Result<()> {
        let matrix = BarkFineCodeMatrix {
            codebooks: vec![vec![1, 2, 3], vec![4, 5, 6]],
        };

        let frames = fine_input_frames(&matrix, 1, 3, 10)?;

        assert_eq!(frames, vec![vec![2, 5], vec![3, 6], vec![10, 10]]);
        Ok(())
    }

    #[test]
    fn fine_loop_count_matches_transformers_window_rule() {
        assert_eq!(fine_generation_loop_count(2, 4, 8), 1);
        assert_eq!(fine_generation_loop_count(10, 4, 8), 2);
        assert_eq!(fine_generation_loop_count(17, 4, 8), 4);
    }

    fn generation_config() -> BarkGenerationConfig {
        BarkGenerationConfig {
            sample_rate: 24_000,
            codebook_size: 1024,
            semantic_config: BarkSemanticGenerationConfig {
                eos_token_id: 10_000,
                max_input_semantic_length: 256,
                max_new_tokens: 768,
                semantic_infer_token: 129_599,
                semantic_pad_token: 10_000,
                semantic_rate_hz: 49.9,
                semantic_vocab_size: 10_000,
                text_encoding_offset: 10_048,
                text_pad_token: 129_595,
                temperature: 0.7,
                top_k: 50,
                top_p: 1.0,
            },
            coarse_acoustics_config: BarkCoarseGenerationConfig {
                coarse_infer_token: 12_050,
                coarse_rate_hz: 75,
                coarse_semantic_pad_token: 12_048,
                max_coarse_history: 630,
                max_coarse_input_length: 256,
                n_coarse_codebooks: 2,
                sliding_window_len: 60,
                temperature: 0.7,
                top_k: 50,
                top_p: 1.0,
            },
            fine_acoustics_config: BarkFineGenerationConfig {
                max_fine_history_length: 512,
                max_fine_input_length: 1024,
                n_fine_codebooks: 8,
                temperature: 0.5,
                top_k: 50,
                top_p: 1.0,
            },
            model_type: Some("bark".to_string()),
        }
    }
}
