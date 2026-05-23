use crate::{models::cpu, runtime::thread_pool::ThreadPool};
use gemm::Parallelism;

use super::{
    add_in_place, embedding_lookup, gelu_in_place, layer_norm_in_place,
    BarkCausalTransformerWeights, BarkError, BarkSubModelConfig, Result,
};

const BARK_LAYER_NORM_EPS: f32 = 1e-5;
const BARK_DEBUG_ENV: &str = "PUPPYGRAD_BARK_DEBUG";
const BARK_DENSE_PARALLEL_THRESHOLD: usize = 262_144;
const BARK_ATTENTION_PARALLEL_THRESHOLD: usize = 262_144;
const BARK_GEMM_THRESHOLD: usize = 8_000_000;

#[derive(Clone, Debug, PartialEq)]
pub struct BarkCausalTransformer {
    pub config: BarkSubModelConfig,
    pub weights: BarkCausalTransformerWeights,
    causal_attention: bool,
    thread_pool: ThreadPool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkKvCache {
    layers: Vec<BarkLayerKvCache>,
    len: usize,
    max_len: usize,
    hidden_size: usize,
    scratch: BarkKvScratch,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct BarkLayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl BarkLayerKvCache {
    fn new(max_len: usize, hidden_size: usize) -> Self {
        Self {
            keys: vec![0.0; max_len * hidden_size],
            values: vec![0.0; max_len * hidden_size],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct BarkKvScratch {
    values: Vec<f32>,
    residual: Vec<f32>,
    attention: Vec<f32>,
    mlp: Vec<f32>,
    qkv: Vec<f32>,
    context: Vec<f32>,
    scores: Vec<f32>,
    last_hidden: Vec<f32>,
    logits: Vec<f32>,
}

impl BarkKvCache {
    pub fn clear(&mut self) {
        self.len = 0;
        self.scratch.values.clear();
        self.scratch.residual.clear();
        self.scratch.attention.clear();
        self.scratch.mlp.clear();
        self.scratch.qkv.clear();
        self.scratch.context.clear();
        self.scratch.scores.clear();
        self.scratch.last_hidden.clear();
        self.scratch.logits.clear();
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl BarkCausalTransformer {
    pub fn new(config: BarkSubModelConfig, weights: BarkCausalTransformerWeights) -> Result<Self> {
        Self::new_with_threads(config, weights, 1)
    }

    pub fn new_with_threads(
        config: BarkSubModelConfig,
        weights: BarkCausalTransformerWeights,
        threads: usize,
    ) -> Result<Self> {
        Self::new_with_attention(config, weights, true, threads)
    }

    pub fn new_bidirectional(
        config: BarkSubModelConfig,
        weights: BarkCausalTransformerWeights,
    ) -> Result<Self> {
        Self::new_bidirectional_with_threads(config, weights, 1)
    }

    pub fn new_bidirectional_with_threads(
        config: BarkSubModelConfig,
        weights: BarkCausalTransformerWeights,
        threads: usize,
    ) -> Result<Self> {
        Self::new_with_attention(config, weights, false, threads)
    }

    fn new_with_attention(
        config: BarkSubModelConfig,
        weights: BarkCausalTransformerWeights,
        causal_attention: bool,
        threads: usize,
    ) -> Result<Self> {
        config.validate("bark_causal_transformer")?;
        if weights.layers.len() != config.num_layers {
            return Err(BarkError::InvalidInput(format!(
                "weight layer count {} does not match config layer count {}",
                weights.layers.len(),
                config.num_layers
            )));
        }
        Ok(Self {
            config,
            weights,
            causal_attention,
            thread_pool: ThreadPool::new(threads),
        })
    }

    pub fn forward_logits(&self, input_ids: &[usize]) -> Result<Vec<f32>> {
        self.forward_logits_with_attention_mask(input_ids, None)
    }

    pub fn forward_logits_with_attention_mask(
        &self,
        input_ids: &[usize],
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(BarkError::InvalidInput(
                "transformer input ids must not be empty".to_string(),
            ));
        }
        if input_ids.len() > self.config.block_size {
            return Err(BarkError::InvalidInput(format!(
                "input length {} exceeds block_size {}",
                input_ids.len(),
                self.config.block_size
            )));
        }

        let values = embedding_lookup(
            input_ids,
            &self.weights.token_embedding,
            self.config.input_vocab_size,
            self.config.hidden_size,
        )?;
        self.forward_logits_from_embeddings_with_attention_mask(
            &values,
            input_ids.len(),
            0,
            attention_mask,
        )
    }

    pub fn forward_logits_from_embeddings(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        position_offset: usize,
    ) -> Result<Vec<f32>> {
        self.forward_logits_from_embeddings_with_attention_mask(
            input_embeddings,
            seq_len,
            position_offset,
            None,
        )
    }

    pub fn forward_logits_from_embeddings_with_attention_mask(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        position_offset: usize,
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let values = self.forward_hidden_from_embeddings_with_attention_mask(
            input_embeddings,
            seq_len,
            position_offset,
            attention_mask,
        )?;
        let logits = linear_parallel(
            &values,
            seq_len,
            self.config.hidden_size,
            &self.weights.lm_head_weight,
            None,
            self.config.output_vocab_size,
            &self.thread_pool,
        )?;
        maybe_dump_debug_logits(
            seq_len,
            self.config.hidden_size,
            self.config.output_vocab_size,
            &logits,
        );
        Ok(logits)
    }

    pub fn forward_hidden_from_embeddings_with_attention_mask(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        position_offset: usize,
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        if seq_len == 0 {
            return Err(BarkError::InvalidInput(
                "transformer input embeddings must not be empty".to_string(),
            ));
        }
        if input_embeddings.len() != seq_len * hidden {
            return Err(BarkError::InvalidInput(format!(
                "input embedding length {} does not match seq_len {seq_len} x hidden_size {hidden}",
                input_embeddings.len()
            )));
        }
        if position_offset + seq_len > self.config.block_size {
            return Err(BarkError::InvalidInput(format!(
                "position range {}..{} exceeds block_size {}",
                position_offset,
                position_offset + seq_len,
                self.config.block_size
            )));
        }
        if attention_mask.is_some_and(|mask| mask.len() != seq_len) {
            return Err(BarkError::InvalidInput(format!(
                "attention mask length does not match seq_len {seq_len}"
            )));
        }
        let mut values = input_embeddings.to_vec();
        for pos in 0..seq_len {
            let pos_start = pos * hidden;
            let embedding_pos_start = (position_offset + pos) * hidden;
            for dim in 0..hidden {
                values[pos_start + dim] +=
                    self.weights.positional_embedding[embedding_pos_start + dim];
            }
        }

        for layer in &self.weights.layers {
            let residual = values.clone();
            layer_norm_in_place(
                &mut values,
                seq_len,
                hidden,
                &layer.attention_layer_norm_weight,
                layer.attention_layer_norm_bias.as_deref(),
                BARK_LAYER_NORM_EPS,
            )?;
            let attention = self_attention_with_mask_parallel(
                &values,
                seq_len,
                hidden,
                self.config.num_heads,
                &layer.attention_qkv_weight,
                layer.attention_qkv_bias.as_deref(),
                &layer.attention_output_weight,
                layer.attention_output_bias.as_deref(),
                self.causal_attention,
                attention_mask,
                &self.thread_pool,
            )?;
            values = residual;
            add_in_place(&mut values, &attention)?;

            let residual = values.clone();
            layer_norm_in_place(
                &mut values,
                seq_len,
                hidden,
                &layer.mlp_layer_norm_weight,
                layer.mlp_layer_norm_bias.as_deref(),
                BARK_LAYER_NORM_EPS,
            )?;
            let mut mlp = linear_parallel(
                &values,
                seq_len,
                hidden,
                &layer.mlp_in_weight,
                layer.mlp_in_bias.as_deref(),
                hidden * 4,
                &self.thread_pool,
            )?;
            gelu_in_place(&mut mlp);
            let mlp = linear_parallel(
                &mlp,
                seq_len,
                hidden * 4,
                &layer.mlp_out_weight,
                layer.mlp_out_bias.as_deref(),
                hidden,
                &self.thread_pool,
            )?;
            values = residual;
            add_in_place(&mut values, &mlp)?;
        }

        layer_norm_in_place(
            &mut values,
            seq_len,
            hidden,
            &self.weights.final_layer_norm_weight,
            self.weights.final_layer_norm_bias.as_deref(),
            BARK_LAYER_NORM_EPS,
        )?;
        Ok(values)
    }

    pub fn logits_from_hidden_with_head(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        lm_head_weight: &[f32],
    ) -> Result<Vec<f32>> {
        linear_parallel(
            hidden_states,
            seq_len,
            self.config.hidden_size,
            lm_head_weight,
            None,
            self.config.output_vocab_size,
            &self.thread_pool,
        )
    }

    pub fn last_logits(&self, input_ids: &[usize]) -> Result<Vec<f32>> {
        let logits = self.forward_logits(input_ids)?;
        self.last_logits_from_full_logits(&logits)
    }

    pub fn last_logits_with_attention_mask(
        &self,
        input_ids: &[usize],
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let logits = self.forward_logits_with_attention_mask(input_ids, attention_mask)?;
        self.last_logits_from_full_logits(&logits)
    }

    pub fn last_logits_from_embeddings(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        position_offset: usize,
    ) -> Result<Vec<f32>> {
        let logits =
            self.forward_logits_from_embeddings(input_embeddings, seq_len, position_offset)?;
        self.last_logits_from_full_logits(&logits)
    }

    pub fn last_logits_from_embeddings_with_attention_mask(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        position_offset: usize,
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let logits = self.forward_logits_from_embeddings_with_attention_mask(
            input_embeddings,
            seq_len,
            position_offset,
            attention_mask,
        )?;
        self.last_logits_from_full_logits(&logits)
    }

    pub fn new_kv_cache(&self) -> BarkKvCache {
        BarkKvCache {
            layers: (0..self.config.num_layers)
                .map(|_| BarkLayerKvCache::new(self.config.block_size, self.config.hidden_size))
                .collect(),
            len: 0,
            max_len: self.config.block_size,
            hidden_size: self.config.hidden_size,
            scratch: BarkKvScratch::default(),
        }
    }

    pub fn cached_last_logits(
        &self,
        input_ids: &[usize],
        cache: &mut BarkKvCache,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(BarkError::InvalidInput(
                "cached transformer input ids must not be empty".to_string(),
            ));
        }
        let values = embedding_lookup(
            input_ids,
            &self.weights.token_embedding,
            self.config.input_vocab_size,
            self.config.hidden_size,
        )?;
        self.cached_last_logits_from_embeddings(&values, input_ids.len(), cache)
    }

    pub fn cached_last_logits_from_embeddings(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        cache: &mut BarkKvCache,
    ) -> Result<Vec<f32>> {
        if !self.causal_attention {
            return Err(BarkError::InvalidInput(
                "cached decoding requires causal attention".to_string(),
            ));
        }
        validate_cache(cache, self.config.num_layers, self.config.hidden_size)?;
        let hidden = self.config.hidden_size;
        if seq_len == 0 {
            return Err(BarkError::InvalidInput(
                "cached transformer input embeddings must not be empty".to_string(),
            ));
        }
        if input_embeddings.len() != seq_len * hidden {
            return Err(BarkError::InvalidInput(format!(
                "input embedding length {} does not match seq_len {seq_len} x hidden_size {hidden}",
                input_embeddings.len()
            )));
        }
        if cache.len + seq_len > self.config.block_size {
            return Err(BarkError::InvalidInput(format!(
                "cached position range {}..{} exceeds block_size {}",
                cache.len,
                cache.len + seq_len,
                self.config.block_size
            )));
        }

        for pos in 0..seq_len {
            let position = cache.len;
            cache.scratch.values.clear();
            cache
                .scratch
                .values
                .extend_from_slice(&input_embeddings[pos * hidden..(pos + 1) * hidden]);
            let embedding_pos_start = position * hidden;
            for dim in 0..hidden {
                cache.scratch.values[dim] +=
                    self.weights.positional_embedding[embedding_pos_start + dim];
            }
            for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
                cache.scratch.residual.clear();
                cache
                    .scratch
                    .residual
                    .extend_from_slice(&cache.scratch.values);
                layer_norm_in_place(
                    &mut cache.scratch.values,
                    1,
                    hidden,
                    &layer.attention_layer_norm_weight,
                    layer.attention_layer_norm_bias.as_deref(),
                    BARK_LAYER_NORM_EPS,
                )?;
                cached_causal_attention_into(
                    &cache.scratch.values,
                    hidden,
                    self.config.num_heads,
                    position,
                    &layer.attention_qkv_weight,
                    layer.attention_qkv_bias.as_deref(),
                    &layer.attention_output_weight,
                    layer.attention_output_bias.as_deref(),
                    &mut cache.layers[layer_idx],
                    &mut cache.scratch.qkv,
                    &mut cache.scratch.context,
                    &mut cache.scratch.scores,
                    &mut cache.scratch.attention,
                    &self.thread_pool,
                )?;
                cache.scratch.values.clear();
                cache
                    .scratch
                    .values
                    .extend_from_slice(&cache.scratch.residual);
                add_in_place(&mut cache.scratch.values, &cache.scratch.attention)?;

                cache.scratch.residual.clear();
                cache
                    .scratch
                    .residual
                    .extend_from_slice(&cache.scratch.values);
                layer_norm_in_place(
                    &mut cache.scratch.values,
                    1,
                    hidden,
                    &layer.mlp_layer_norm_weight,
                    layer.mlp_layer_norm_bias.as_deref(),
                    BARK_LAYER_NORM_EPS,
                )?;
                linear_into_parallel(
                    &cache.scratch.values,
                    1,
                    hidden,
                    &layer.mlp_in_weight,
                    layer.mlp_in_bias.as_deref(),
                    hidden * 4,
                    &mut cache.scratch.mlp,
                    &self.thread_pool,
                )?;
                gelu_in_place(&mut cache.scratch.mlp);
                linear_into_parallel(
                    &cache.scratch.mlp,
                    1,
                    hidden * 4,
                    &layer.mlp_out_weight,
                    layer.mlp_out_bias.as_deref(),
                    hidden,
                    &mut cache.scratch.attention,
                    &self.thread_pool,
                )?;
                cache.scratch.values.clear();
                cache
                    .scratch
                    .values
                    .extend_from_slice(&cache.scratch.residual);
                add_in_place(&mut cache.scratch.values, &cache.scratch.attention)?;
            }

            layer_norm_in_place(
                &mut cache.scratch.values,
                1,
                hidden,
                &self.weights.final_layer_norm_weight,
                self.weights.final_layer_norm_bias.as_deref(),
                BARK_LAYER_NORM_EPS,
            )?;
            cache.len += 1;
            cache.scratch.last_hidden.clear();
            cache
                .scratch
                .last_hidden
                .extend_from_slice(&cache.scratch.values);
        }

        linear_into_parallel(
            &cache.scratch.last_hidden,
            1,
            hidden,
            &self.weights.lm_head_weight,
            None,
            self.config.output_vocab_size,
            &mut cache.scratch.logits,
            &self.thread_pool,
        )?;
        Ok(cache.scratch.logits.clone())
    }

    pub fn prefill_cached_last_logits_from_embeddings(
        &self,
        input_embeddings: &[f32],
        seq_len: usize,
        cache: &mut BarkKvCache,
    ) -> Result<Vec<f32>> {
        if !self.causal_attention {
            return Err(BarkError::InvalidInput(
                "cached prefill requires causal attention".to_string(),
            ));
        }
        validate_cache(cache, self.config.num_layers, self.config.hidden_size)?;
        let hidden = self.config.hidden_size;
        if seq_len == 0 {
            return Err(BarkError::InvalidInput(
                "cached prefill input embeddings must not be empty".to_string(),
            ));
        }
        if !cache.is_empty() {
            return Err(BarkError::InvalidInput(
                "cached prefill requires an empty KV cache".to_string(),
            ));
        }
        if input_embeddings.len() != seq_len * hidden {
            return Err(BarkError::InvalidInput(format!(
                "input embedding length {} does not match seq_len {seq_len} x hidden_size {hidden}",
                input_embeddings.len()
            )));
        }
        if seq_len > self.config.block_size {
            return Err(BarkError::InvalidInput(format!(
                "cached prefill length {seq_len} exceeds block_size {}",
                self.config.block_size
            )));
        }

        let mut values = input_embeddings.to_vec();
        for pos in 0..seq_len {
            let pos_start = pos * hidden;
            let embedding_pos_start = pos * hidden;
            for dim in 0..hidden {
                values[pos_start + dim] +=
                    self.weights.positional_embedding[embedding_pos_start + dim];
            }
        }

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let residual = values.clone();
            layer_norm_in_place(
                &mut values,
                seq_len,
                hidden,
                &layer.attention_layer_norm_weight,
                layer.attention_layer_norm_bias.as_deref(),
                BARK_LAYER_NORM_EPS,
            )?;
            let qkv = linear_parallel(
                &values,
                seq_len,
                hidden,
                &layer.attention_qkv_weight,
                layer.attention_qkv_bias.as_deref(),
                hidden * 3,
                &self.thread_pool,
            )?;
            fill_layer_cache_from_qkv(&qkv, seq_len, hidden, &mut cache.layers[layer_idx])?;
            let attention = self_attention_from_qkv_parallel(
                &qkv,
                seq_len,
                hidden,
                self.config.num_heads,
                &layer.attention_output_weight,
                layer.attention_output_bias.as_deref(),
                true,
                None,
                &self.thread_pool,
            )?;
            values = residual;
            add_in_place(&mut values, &attention)?;

            let residual = values.clone();
            layer_norm_in_place(
                &mut values,
                seq_len,
                hidden,
                &layer.mlp_layer_norm_weight,
                layer.mlp_layer_norm_bias.as_deref(),
                BARK_LAYER_NORM_EPS,
            )?;
            let mut mlp = linear_parallel(
                &values,
                seq_len,
                hidden,
                &layer.mlp_in_weight,
                layer.mlp_in_bias.as_deref(),
                hidden * 4,
                &self.thread_pool,
            )?;
            gelu_in_place(&mut mlp);
            let mlp = linear_parallel(
                &mlp,
                seq_len,
                hidden * 4,
                &layer.mlp_out_weight,
                layer.mlp_out_bias.as_deref(),
                hidden,
                &self.thread_pool,
            )?;
            values = residual;
            add_in_place(&mut values, &mlp)?;
        }

        layer_norm_in_place(
            &mut values,
            seq_len,
            hidden,
            &self.weights.final_layer_norm_weight,
            self.weights.final_layer_norm_bias.as_deref(),
            BARK_LAYER_NORM_EPS,
        )?;
        cache.len = seq_len;
        cache.scratch.last_hidden.clear();
        cache
            .scratch
            .last_hidden
            .extend_from_slice(&values[(seq_len - 1) * hidden..seq_len * hidden]);
        linear_into_parallel(
            &cache.scratch.last_hidden,
            1,
            hidden,
            &self.weights.lm_head_weight,
            None,
            self.config.output_vocab_size,
            &mut cache.scratch.logits,
            &self.thread_pool,
        )?;
        Ok(cache.scratch.logits.clone())
    }

    fn last_logits_from_full_logits(&self, logits: &[f32]) -> Result<Vec<f32>> {
        let vocab = self.config.output_vocab_size;
        Ok(logits[logits.len() - vocab..].to_vec())
    }
}

fn fill_layer_cache_from_qkv(
    qkv: &[f32],
    seq_len: usize,
    hidden: usize,
    cache: &mut BarkLayerKvCache,
) -> Result<()> {
    if qkv.len() != seq_len * hidden * 3 || cache.keys.len() < seq_len * hidden {
        return Err(BarkError::InvalidInput(
            "cached prefill qkv/cache shape mismatch".to_string(),
        ));
    }
    for pos in 0..seq_len {
        let qkv_start = pos * hidden * 3;
        let cache_start = pos * hidden;
        cache.keys[cache_start..cache_start + hidden]
            .copy_from_slice(&qkv[qkv_start + hidden..qkv_start + hidden * 2]);
        cache.values[cache_start..cache_start + hidden]
            .copy_from_slice(&qkv[qkv_start + hidden * 2..qkv_start + hidden * 3]);
    }
    Ok(())
}

fn validate_cache(cache: &BarkKvCache, layers: usize, hidden: usize) -> Result<()> {
    if cache.layers.len() != layers {
        return Err(BarkError::InvalidInput(format!(
            "KV cache layer count {} does not match transformer layer count {layers}",
            cache.layers.len()
        )));
    }
    if cache.hidden_size != hidden {
        return Err(BarkError::InvalidInput(format!(
            "KV cache hidden size {} does not match transformer hidden size {hidden}",
            cache.hidden_size
        )));
    }
    if cache.len > cache.max_len {
        return Err(BarkError::InvalidInput(format!(
            "KV cache length {} exceeds max length {}",
            cache.len, cache.max_len
        )));
    }
    for (idx, layer) in cache.layers.iter().enumerate() {
        if layer.keys.len() != cache.max_len * hidden
            || layer.values.len() != cache.max_len * hidden
        {
            return Err(BarkError::InvalidInput(format!(
                "KV cache layer {idx} has inconsistent key/value shapes for max length {} and hidden size {hidden}",
                cache.max_len
            )));
        }
    }
    Ok(())
}

fn linear_parallel(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    pool: &ThreadPool,
) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    linear_into_parallel(
        input,
        rows,
        in_features,
        weight,
        bias,
        out_features,
        &mut out,
        pool,
    )?;
    Ok(out)
}

fn linear_into_parallel(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    out: &mut Vec<f32>,
    pool: &ThreadPool,
) -> Result<()> {
    if input.len() != rows * in_features {
        return Err(BarkError::InvalidInput(format!(
            "linear input length {} does not match rows {rows} x in_features {in_features}",
            input.len()
        )));
    }
    if weight.len() != out_features * in_features {
        return Err(BarkError::InvalidInput(format!(
            "linear weight length {} does not match out_features {out_features} x in_features {in_features}",
            weight.len()
        )));
    }
    if bias.is_some_and(|bias| bias.len() != out_features) {
        return Err(BarkError::InvalidInput(
            "linear bias length does not match out features".to_string(),
        ));
    }
    out.clear();
    let work_items = rows * in_features * out_features;
    if work_items >= BARK_GEMM_THRESHOLD {
        let zero_bias;
        let bias = match bias {
            Some(bias) => bias,
            None => {
                zero_bias = vec![0.0; out_features];
                &zero_bias
            }
        };
        gemm_transposed_dense_projection_into(
            input,
            rows,
            in_features,
            weight,
            bias,
            out_features,
            out,
            pool.threads(),
        );
        return Ok(());
    }
    if pool.threads() == 1 || rows * in_features * out_features < BARK_DENSE_PARALLEL_THRESHOLD {
        let zero_bias;
        let bias = match bias {
            Some(bias) => bias,
            None => {
                zero_bias = vec![0.0; out_features];
                &zero_bias
            }
        };
        cpu::transposed_dense_projection_into(
            input,
            cpu::DenseShape::new(rows, in_features, out_features),
            weight,
            bias,
            out,
        );
        return Ok(());
    }

    if pool.threads() > 1 && rows * in_features * out_features >= BARK_DENSE_PARALLEL_THRESHOLD {
        gemm_linear_into(
            input,
            rows,
            in_features,
            weight,
            bias,
            out_features,
            out,
            pool.threads(),
        );
        return Ok(());
    }

    Ok(())
}

fn gemm_linear_into(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    out: &mut Vec<f32>,
    threads: usize,
) {
    out.clear();
    out.resize(rows * out_features, 0.0);
    unsafe {
        gemm::gemm(
            rows,
            out_features,
            in_features,
            out.as_mut_ptr(),
            1,
            out_features as isize,
            false,
            input.as_ptr(),
            1,
            in_features as isize,
            weight.as_ptr(),
            in_features as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            Parallelism::Rayon(threads),
        );
    }
    if let Some(bias) = bias {
        for row in out.chunks_mut(out_features) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += *bias;
            }
        }
    }
}

fn gemm_transposed_dense_projection_into(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    out_features: usize,
    out: &mut Vec<f32>,
    threads: usize,
) {
    out.clear();
    out.resize(rows * out_features, 0.0);
    unsafe {
        gemm::gemm(
            rows,
            out_features,
            in_features,
            out.as_mut_ptr(),
            1,
            out_features as isize,
            false,
            input.as_ptr(),
            1,
            in_features as isize,
            weight.as_ptr(),
            in_features as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            Parallelism::Rayon(threads.max(1)),
        );
    }
    for row in 0..rows {
        let start = row * out_features;
        for col in 0..out_features {
            out[start + col] += bias[col];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention_with_mask_parallel(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
    causal: bool,
    attention_mask: Option<&[bool]>,
    pool: &ThreadPool,
) -> Result<Vec<f32>> {
    if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
        return Err(BarkError::InvalidInput(
            "hidden size must be divisible by num_heads".to_string(),
        ));
    }
    if attention_mask.is_some_and(|mask| mask.len() != seq_len) {
        return Err(BarkError::InvalidInput(format!(
            "attention mask length does not match seq_len {seq_len}"
        )));
    }
    let qkv = linear_parallel(
        input,
        seq_len,
        hidden_size,
        qkv_weight,
        qkv_bias,
        hidden_size * 3,
        pool,
    )?;
    self_attention_from_qkv_parallel(
        &qkv,
        seq_len,
        hidden_size,
        num_heads,
        out_weight,
        out_bias,
        causal,
        attention_mask,
        pool,
    )
}

#[allow(clippy::too_many_arguments)]
fn self_attention_from_qkv_parallel(
    qkv: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
    causal: bool,
    attention_mask: Option<&[bool]>,
    pool: &ThreadPool,
) -> Result<Vec<f32>> {
    if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
        return Err(BarkError::InvalidInput(
            "hidden size must be divisible by num_heads".to_string(),
        ));
    }
    if qkv.len() != seq_len * hidden_size * 3 {
        return Err(BarkError::InvalidInput(
            "attention qkv shape mismatch".to_string(),
        ));
    }
    if attention_mask.is_some_and(|mask| mask.len() != seq_len) {
        return Err(BarkError::InvalidInput(format!(
            "attention mask length does not match seq_len {seq_len}"
        )));
    }
    let head_dim = hidden_size / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let work_items = seq_len * seq_len * num_heads * head_dim;
    let mut context = vec![0.0; seq_len * hidden_size];

    if pool.threads() > 1 && work_items >= BARK_ATTENTION_PARALLEL_THRESHOLD {
        let heads = pool.scoped_parallel_chunks(num_heads, 1, |start, end| {
            let mut head_outputs = Vec::with_capacity(end - start);
            for head in start..end {
                let mut head_out = vec![0.0; seq_len * head_dim];
                let mut scores = vec![0.0; seq_len];
                for pos in 0..seq_len {
                    let attention_len = if causal { pos + 1 } else { seq_len };
                    let q_base = pos * hidden_size * 3 + head * head_dim;
                    for key_pos in 0..attention_len {
                        let k_base = key_pos * hidden_size * 3 + hidden_size + head * head_dim;
                        let mut score = 0.0;
                        for dim in 0..head_dim {
                            score += qkv[q_base + dim] * qkv[k_base + dim];
                        }
                        scores[key_pos] = if attention_mask.is_some_and(|mask| !mask[key_pos]) {
                            f32::NEG_INFINITY
                        } else {
                            score * scale
                        };
                    }
                    let has_attention = softmax_prefix_in_place(&mut scores, attention_len);
                    if !has_attention {
                        continue;
                    }
                    for dim in 0..head_dim {
                        let mut value = 0.0;
                        for key_pos in 0..attention_len {
                            let v_base =
                                key_pos * hidden_size * 3 + hidden_size * 2 + head * head_dim;
                            value += scores[key_pos] * qkv[v_base + dim];
                        }
                        head_out[pos * head_dim + dim] = value;
                    }
                }
                head_outputs.push((head, head_out));
            }
            head_outputs
        });

        for (head, head_out) in heads.into_iter().flatten() {
            for pos in 0..seq_len {
                let dst_start = pos * hidden_size + head * head_dim;
                context[dst_start..dst_start + head_dim]
                    .copy_from_slice(&head_out[pos * head_dim..(pos + 1) * head_dim]);
            }
        }
    } else {
        let mut scores = vec![0.0; seq_len];
        for pos in 0..seq_len {
            let attention_len = if causal { pos + 1 } else { seq_len };
            for head in 0..num_heads {
                let q_base = pos * hidden_size * 3 + head * head_dim;
                for key_pos in 0..attention_len {
                    let k_base = key_pos * hidden_size * 3 + hidden_size + head * head_dim;
                    let mut score = 0.0;
                    for dim in 0..head_dim {
                        score += qkv[q_base + dim] * qkv[k_base + dim];
                    }
                    scores[key_pos] = if attention_mask.is_some_and(|mask| !mask[key_pos]) {
                        f32::NEG_INFINITY
                    } else {
                        score * scale
                    };
                }
                let has_attention = softmax_prefix_in_place(&mut scores, attention_len);
                if !has_attention {
                    continue;
                }
                for dim in 0..head_dim {
                    let mut value = 0.0;
                    for key_pos in 0..attention_len {
                        let v_base = key_pos * hidden_size * 3 + hidden_size * 2 + head * head_dim;
                        value += scores[key_pos] * qkv[v_base + dim];
                    }
                    context[pos * hidden_size + head * head_dim + dim] = value;
                }
            }
        }
    }

    linear_parallel(
        &context,
        seq_len,
        hidden_size,
        out_weight,
        out_bias,
        hidden_size,
        pool,
    )
}

fn cached_causal_attention_into(
    input: &[f32],
    hidden_size: usize,
    num_heads: usize,
    position: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
    cache: &mut BarkLayerKvCache,
    qkv: &mut Vec<f32>,
    context: &mut Vec<f32>,
    scores: &mut Vec<f32>,
    out: &mut Vec<f32>,
    pool: &ThreadPool,
) -> Result<()> {
    if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
        return Err(BarkError::InvalidInput(
            "hidden size must be divisible by num_heads".to_string(),
        ));
    }
    if cache.keys.len() != cache.values.len()
        || cache.keys.len() < (position + 1) * hidden_size
        || input.len() != hidden_size
    {
        return Err(BarkError::InvalidInput(
            "cached attention input/cache shape mismatch".to_string(),
        ));
    }
    linear_into_parallel(
        input,
        1,
        hidden_size,
        qkv_weight,
        qkv_bias,
        hidden_size * 3,
        qkv,
        pool,
    )?;
    let query = &qkv[..hidden_size];
    let key = &qkv[hidden_size..hidden_size * 2];
    let value = &qkv[hidden_size * 2..hidden_size * 3];
    let cache_start = position * hidden_size;
    cache.keys[cache_start..cache_start + hidden_size].copy_from_slice(key);
    cache.values[cache_start..cache_start + hidden_size].copy_from_slice(value);

    let head_dim = hidden_size / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attention_len = position + 1;
    context.clear();
    context.resize(hidden_size, 0.0);
    scores.clear();
    scores.resize(attention_len, 0.0);
    let work_items = attention_len * num_heads * head_dim;
    if pool.threads() > 1 && work_items >= BARK_ATTENTION_PARALLEL_THRESHOLD {
        let heads = pool.scoped_parallel_chunks(num_heads, 1, |start, end| {
            let mut head_outputs = Vec::with_capacity(end - start);
            for head in start..end {
                let q_base = head * head_dim;
                let mut local_scores = vec![0.0; attention_len];
                for key_pos in 0..attention_len {
                    let k_base = key_pos * hidden_size + head * head_dim;
                    let mut score = 0.0;
                    for dim in 0..head_dim {
                        score += query[q_base + dim] * cache.keys[k_base + dim];
                    }
                    local_scores[key_pos] = score * scale;
                }
                softmax_prefix_in_place(&mut local_scores, attention_len);
                let mut head_out = vec![0.0; head_dim];
                for dim in 0..head_dim {
                    let mut weighted_value = 0.0;
                    for key_pos in 0..attention_len {
                        let v_base = key_pos * hidden_size + head * head_dim;
                        weighted_value += local_scores[key_pos] * cache.values[v_base + dim];
                    }
                    head_out[dim] = weighted_value;
                }
                head_outputs.push((head, head_out));
            }
            head_outputs
        });
        for (head, head_out) in heads.into_iter().flatten() {
            let dst_start = head * head_dim;
            context[dst_start..dst_start + head_dim].copy_from_slice(&head_out);
        }
    } else {
        for head in 0..num_heads {
            let q_base = head * head_dim;
            for key_pos in 0..attention_len {
                let k_base = key_pos * hidden_size + head * head_dim;
                let mut score = 0.0;
                for dim in 0..head_dim {
                    score += query[q_base + dim] * cache.keys[k_base + dim];
                }
                scores[key_pos] = score * scale;
            }
            softmax_prefix_in_place(scores, attention_len);
            for dim in 0..head_dim {
                let mut weighted_value = 0.0;
                for key_pos in 0..attention_len {
                    let v_base = key_pos * hidden_size + head * head_dim;
                    weighted_value += scores[key_pos] * cache.values[v_base + dim];
                }
                context[head * head_dim + dim] = weighted_value;
            }
        }
    }
    linear_into_parallel(
        context,
        1,
        hidden_size,
        out_weight,
        out_bias,
        hidden_size,
        out,
        pool,
    )
}

fn softmax_prefix_in_place(values: &mut [f32], len: usize) -> bool {
    let max = values[..len]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        for value in &mut values[..len] {
            *value = 0.0;
        }
        return false;
    }
    let mut total = 0.0;
    for value in &mut values[..len] {
        *value = (*value - max).exp();
        total += *value;
    }
    if total > 0.0 && total.is_finite() {
        for value in &mut values[..len] {
            *value /= total;
        }
        true
    } else {
        false
    }
}

fn maybe_dump_debug_logits(seq_len: usize, hidden: usize, vocab: usize, logits: &[f32]) {
    if std::env::var_os(BARK_DEBUG_ENV).is_none() {
        return;
    }
    let start = logits.len().saturating_sub(vocab);
    let slice_len = vocab.min(8);
    eprintln!(
        "bark-transformer seq_len={seq_len} hidden={hidden} vocab={vocab} last_logits_head={:?}",
        &logits[start..start + slice_len]
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bark::{BarkCausalTransformerLayerWeights, BarkCausalTransformerWeights};

    #[test]
    fn transformer_forward_returns_logits_for_each_input_token() -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config.clone(), weights)?;

        let logits = model.forward_logits(&[0, 1])?;
        let last = model.last_logits(&[0, 1])?;

        assert_eq!(logits.len(), 2 * config.output_vocab_size);
        assert_eq!(last.len(), config.output_vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn cached_causal_forward_matches_full_forward_last_logits() -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config.clone(), weights)?;
        let mut cache = model.new_kv_cache();
        let key_capacity = cache.layers[0].keys.len();
        let value_capacity = cache.layers[0].values.len();

        let first = model.cached_last_logits(&[0, 1], &mut cache)?;
        let full_first = model.last_logits(&[0, 1])?;
        assert_eq!(first, full_first);
        assert_eq!(cache.layers[0].keys.len(), key_capacity);
        assert_eq!(cache.layers[0].values.len(), value_capacity);

        let second = model.cached_last_logits(&[2], &mut cache)?;
        let full_second = model.last_logits(&[0, 1, 2])?;
        assert_eq!(second, full_second);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.layers[0].keys.len(), key_capacity);
        assert_eq!(cache.layers[0].values.len(), value_capacity);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.layers[0].keys.len(), key_capacity);
        assert_eq!(cache.layers[0].values.len(), value_capacity);
        let replay = model.cached_last_logits(&[0, 1, 2], &mut cache)?;
        assert_eq!(replay, full_second);
        Ok(())
    }

    #[test]
    fn cached_causal_forward_rejects_bidirectional_and_bad_shapes() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let bidirectional =
            BarkCausalTransformer::new_bidirectional(config.clone(), weights.clone()).unwrap();
        let mut cache = bidirectional.new_kv_cache();
        let err = bidirectional
            .cached_last_logits(&[0], &mut cache)
            .unwrap_err();
        assert!(err.to_string().contains("causal attention"));

        let model = BarkCausalTransformer::new(config, weights).unwrap();
        let mut cache = model.new_kv_cache();
        let err = model
            .cached_last_logits_from_embeddings(&[1.0, 0.0, 0.0], 2, &mut cache)
            .unwrap_err();
        assert!(err.to_string().contains("input embedding length"));
    }

    #[test]
    fn transformer_forward_accepts_prebuilt_input_embeddings() -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config.clone(), weights)?;

        let logits = model.forward_logits_from_embeddings(&[1.0, 0.0, 0.0, 1.0], 2, 0)?;

        assert_eq!(logits.len(), 2 * config.output_vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn transformer_forward_accepts_attention_mask() -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config.clone(), weights)?;

        let logits = model.forward_logits_with_attention_mask(&[0, 1], Some(&[true, false]))?;

        assert_eq!(logits.len(), 2 * config.output_vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn transformer_rejects_wrong_attention_mask_length() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config, weights).unwrap();

        let err = model
            .forward_logits_with_attention_mask(&[0, 1], Some(&[true]))
            .unwrap_err();

        assert!(err.to_string().contains("attention mask length"));
    }

    #[test]
    fn transformer_rejects_empty_and_out_of_vocab_inputs() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config, weights).unwrap();

        let empty_err = model.forward_logits(&[]).unwrap_err();
        assert!(empty_err.to_string().contains("must not be empty"));

        let vocab_err = model.forward_logits(&[4]).unwrap_err();
        assert!(vocab_err.to_string().contains("outside vocab size"));
    }

    #[test]
    fn transformer_rejects_bad_embedding_shapes_and_positions() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config, weights).unwrap();

        let shape_err = model
            .forward_logits_from_embeddings(&[1.0, 0.0, 0.0], 2, 0)
            .unwrap_err();
        assert!(shape_err.to_string().contains("input embedding length"));

        let position_err = model
            .forward_logits_from_embeddings(&[1.0, 0.0, 0.0, 1.0], 2, 3)
            .unwrap_err();
        assert!(position_err.to_string().contains("position range"));
    }

    #[test]
    fn bidirectional_transformer_uses_full_attention() -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let causal = BarkCausalTransformer::new(config.clone(), weights.clone())?;
        let bidirectional = BarkCausalTransformer::new_bidirectional(config, weights)?;

        let causal_logits = causal.forward_logits(&[0, 1])?;
        let full_logits = bidirectional.forward_logits(&[0, 1])?;

        assert_ne!(causal_logits, full_logits);
        assert!(full_logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn threaded_transformer_matches_single_threaded_forward() -> Result<()> {
        let config = BarkSubModelConfig {
            block_size: 64,
            input_vocab_size: 4,
            output_vocab_size: 128,
            num_layers: 1,
            num_heads: 4,
            hidden_size: 32,
            dropout: 0.0,
            bias: true,
            use_cache: false,
            model_type: Some("semantic".to_string()),
        };
        let weights = tiny_weights(&config);
        let single = BarkCausalTransformer::new_with_threads(config.clone(), weights.clone(), 1)?;
        let threaded = BarkCausalTransformer::new_with_threads(config, weights, 4)?;
        let input = (0..64).map(|idx| idx % 4).collect::<Vec<_>>();

        let single_logits = single.forward_logits(&input)?;
        let threaded_logits = threaded.forward_logits(&input)?;

        assert_eq!(single_logits, threaded_logits);
        Ok(())
    }

    #[test]
    fn transformer_rejects_inputs_longer_than_block_size() -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let model = BarkCausalTransformer::new(config, weights)?;

        let err = model.forward_logits(&[0, 1, 2, 3, 4]).unwrap_err();

        assert!(err.to_string().contains("block_size"));
        Ok(())
    }

    fn tiny_config() -> BarkSubModelConfig {
        BarkSubModelConfig {
            block_size: 4,
            input_vocab_size: 4,
            output_vocab_size: 5,
            num_layers: 1,
            num_heads: 1,
            hidden_size: 2,
            dropout: 0.0,
            bias: true,
            use_cache: false,
            model_type: Some("semantic".to_string()),
        }
    }

    fn tiny_weights(config: &BarkSubModelConfig) -> BarkCausalTransformerWeights {
        let hidden = config.hidden_size;
        let mlp_hidden = hidden * 4;
        let token_embedding = if hidden == 2 && config.input_vocab_size == 4 {
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0, //
                -1.0, 0.5,
            ]
        } else {
            let mut values = vec![0.0; config.input_vocab_size * hidden];
            for token in 0..config.input_vocab_size {
                for dim in 0..hidden {
                    values[token * hidden + dim] = ((token + dim) % 7) as f32 * 0.01;
                }
            }
            values
        };
        BarkCausalTransformerWeights {
            token_embedding,
            positional_embedding: vec![0.0; config.block_size * hidden],
            layers: vec![BarkCausalTransformerLayerWeights {
                attention_layer_norm_weight: vec![1.0; hidden],
                attention_layer_norm_bias: Some(vec![0.0; hidden]),
                attention_qkv_weight: identity_rows(hidden * 3, hidden),
                attention_qkv_bias: Some(vec![0.0; hidden * 3]),
                attention_output_weight: identity_rows(hidden, hidden),
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
            lm_head_weight: identity_rows(config.output_vocab_size, hidden),
        }
    }

    fn identity_rows(rows: usize, cols: usize) -> Vec<f32> {
        let mut values = vec![0.0; rows * cols];
        for row in 0..rows {
            values[row * cols + row % cols] = 1.0;
        }
        values
    }
}
