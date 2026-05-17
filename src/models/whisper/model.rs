use std::time::Duration;

use crate::models::autoregressive::{self, ConditionalAutoregressiveDecoder, KvCache};
use crate::models::cpu::{
    dot, gelu_in_place, layer_norm_in_place, quantized_dot, row, row_mut, softmax_in_place,
    DenseShape, QuantizedRows,
};
use crate::models::generation::{LogitsSampler, TextGenerationConfig};
use crate::runtime::thread_pool::ThreadPool;

use super::weights::{
    WhisperAttentionWeights, WhisperDecoderLayerWeights, WhisperEncoderLayerWeights,
};

#[derive(Clone, Copy)]
struct AttentionShape {
    query_rows: usize,
    key_rows: usize,
    state: usize,
    heads: usize,
    causal: bool,
}
use super::{
    LogMelSpectrogram, Result, WhisperConfig, WhisperError, WhisperPreprocessorConfig, WhisperSize,
    WhisperWeights,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhisperBackendConfig {
    Rust(WhisperRustConfig),
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperBackendName {
    Rust,
    Gpu,
}

impl WhisperBackendConfig {
    pub fn rust(threads: usize, size: WhisperSize) -> Result<Self> {
        let config = WhisperRustConfig::for_size(size).with_threads(threads);
        config.validate()?;
        Ok(Self::Rust(config))
    }

    pub fn name(&self) -> WhisperBackendName {
        match self {
            Self::Rust(_) => WhisperBackendName::Rust,
            Self::Gpu => WhisperBackendName::Gpu,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Rust(config) => format!(
                "rust (threads={}, dense_threshold={}, dense_chunk={}, logits_chunk={}, attention_head_threshold={}, logits_weights={})",
                config.threads,
                config.dense_parallel_threshold,
                config.dense_chunk_size,
                config.logits_chunk_size,
                config.attention_head_parallel_threshold,
                if config.quantized_weights { "int8" } else { "f32" }
            ),
            Self::Gpu => "gpu (not implemented)".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperRustConfig {
    pub threads: usize,
    pub dense_parallel_threshold: usize,
    pub dense_chunk_size: usize,
    pub logits_chunk_size: usize,
    pub attention_head_parallel_threshold: usize,
    pub attention_head_chunk_size: usize,
    pub quantized_weights: bool,
}

impl Default for WhisperRustConfig {
    fn default() -> Self {
        Self {
            threads: 1,
            dense_parallel_threshold: 262_144,
            dense_chunk_size: 64,
            logits_chunk_size: 256,
            attention_head_parallel_threshold: 16_384,
            attention_head_chunk_size: 1,
            quantized_weights: false,
        }
    }
}

impl WhisperRustConfig {
    pub fn for_size(size: WhisperSize) -> Self {
        let mut config = Self::default();
        match size {
            WhisperSize::Tiny | WhisperSize::TinyEn => {
                config.dense_chunk_size = 64;
                config.logits_chunk_size = 256;
                config.attention_head_parallel_threshold = 16_384;
            }
            WhisperSize::Base | WhisperSize::BaseEn => {
                config.dense_chunk_size = 64;
                config.logits_chunk_size = 256;
                config.attention_head_parallel_threshold = 12_288;
            }
            WhisperSize::Small | WhisperSize::SmallEn => {
                config.dense_chunk_size = 48;
                config.logits_chunk_size = 256;
                config.attention_head_parallel_threshold = 8_192;
            }
            WhisperSize::Medium | WhisperSize::MediumEn => {
                config.dense_chunk_size = 32;
                config.logits_chunk_size = 192;
                config.attention_head_parallel_threshold = 4_096;
            }
            WhisperSize::LargeV1 | WhisperSize::LargeV2 | WhisperSize::LargeV3 => {
                config.dense_chunk_size = 32;
                config.logits_chunk_size = 128;
                config.attention_head_parallel_threshold = 4_096;
            }
            WhisperSize::Turbo => {
                config.dense_chunk_size = 32;
                config.logits_chunk_size = 128;
                config.attention_head_parallel_threshold = 4_096;
            }
        }
        config
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_quantized_weights(mut self, quantized_weights: bool) -> Self {
        self.quantized_weights = quantized_weights;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.threads == 0 {
            return Err(WhisperError::InvalidInput(
                "whisper rust backend threads must be > 0".to_string(),
            ));
        }
        if self.dense_parallel_threshold == 0 {
            return Err(WhisperError::InvalidInput(
                "whisper dense parallel threshold must be > 0".to_string(),
            ));
        }
        if self.dense_chunk_size == 0 {
            return Err(WhisperError::InvalidInput(
                "whisper dense chunk size must be > 0".to_string(),
            ));
        }
        if self.logits_chunk_size == 0 {
            return Err(WhisperError::InvalidInput(
                "whisper logits chunk size must be > 0".to_string(),
            ));
        }
        if self.attention_head_parallel_threshold == 0 {
            return Err(WhisperError::InvalidInput(
                "whisper attention head parallel threshold must be > 0".to_string(),
            ));
        }
        if self.attention_head_chunk_size == 0 {
            return Err(WhisperError::InvalidInput(
                "whisper attention head chunk size must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedAudio {
    pub values: Vec<f32>,
    pub frames: usize,
    pub state: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperDecoderKvCache {
    pub layers: Vec<WhisperDecoderLayerKvCache>,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperDecoderLayerKvCache {
    pub self_keys: Vec<f32>,
    pub self_values: Vec<f32>,
}

impl WhisperDecoderKvCache {
    pub fn new(config: &WhisperConfig) -> Self {
        let layer_len = config.n_text_ctx * config.n_text_state;
        Self {
            layers: (0..config.n_text_layer)
                .map(|_| WhisperDecoderLayerKvCache {
                    self_keys: vec![0.0; layer_len],
                    self_values: vec![0.0; layer_len],
                })
                .collect(),
            seq_len: 0,
            max_seq_len: config.n_text_ctx,
        }
    }
}

impl KvCache for WhisperDecoderKvCache {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.layers {
            layer.self_keys.fill(0.0);
            layer.self_values.fill(0.0);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WhisperOperationProfile {
    pub audio_projection: Duration,
    pub encoder_attention: Duration,
    pub encoder_mlp: Duration,
    pub encoder_layer_norm: Duration,
    pub decoder_self_attention: Duration,
    pub decoder_cross_attention: Duration,
    pub decoder_mlp: Duration,
    pub decoder_layer_norm: Duration,
    pub final_logits: Duration,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WhisperForwardScratch {
    norm: Vec<f32>,
    attention: WhisperAttentionScratch,
    attention_out: Vec<f32>,
    hidden: Vec<f32>,
    mlp: Vec<f32>,
    logits: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct WhisperAttentionScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    values: Vec<f32>,
    scores: Vec<f32>,
}

pub fn encode_audio(
    config: &WhisperConfig,
    preprocessor: &WhisperPreprocessorConfig,
    weights: &WhisperWeights,
    features: &LogMelSpectrogram,
    profile: &mut WhisperOperationProfile,
) -> Result<EncodedAudio> {
    let pool = ThreadPool::new(1);
    encode_audio_with_rust_config(
        config,
        preprocessor,
        weights,
        features,
        profile,
        &pool,
        &WhisperRustConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_audio_with_rust_config(
    config: &WhisperConfig,
    preprocessor: &WhisperPreprocessorConfig,
    weights: &WhisperWeights,
    features: &LogMelSpectrogram,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Result<EncodedAudio> {
    let mut scratch = WhisperForwardScratch::default();
    encode_audio_with_scratch(
        config,
        preprocessor,
        weights,
        features,
        profile,
        pool,
        rust_config,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_audio_with_scratch(
    config: &WhisperConfig,
    preprocessor: &WhisperPreprocessorConfig,
    weights: &WhisperWeights,
    features: &LogMelSpectrogram,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperForwardScratch,
) -> Result<EncodedAudio> {
    rust_config.validate()?;
    if features.n_mels != config.n_mels || features.n_frames != preprocessor.n_frames {
        return Err(WhisperError::InvalidInput(format!(
            "log-mel features have shape {}x{}, expected {}x{}",
            features.n_mels, features.n_frames, config.n_mels, preprocessor.n_frames
        )));
    }

    let start = std::time::Instant::now();
    let conv1 = conv1d_features(
        &features.values,
        features.n_mels,
        features.n_frames,
        &weights.encoder.conv1_w,
        &weights.encoder.conv1_b,
        config.n_audio_state,
        1,
        pool,
        rust_config,
    );
    let mut x = conv1;
    gelu_in_place(&mut x);
    let conv2 = conv1d_frame_major(
        &x,
        features.n_frames,
        config.n_audio_state,
        &weights.encoder.conv2_w,
        &weights.encoder.conv2_b,
        config.n_audio_state,
        2,
        pool,
        rust_config,
    );
    x = conv2;
    gelu_in_place(&mut x);
    profile.audio_projection += start.elapsed();

    let frames = config.n_audio_ctx;
    let state = config.n_audio_state;
    for frame in 0..frames {
        let pos = row(&weights.encoder.positional_embedding, frame, state);
        let dst = row_mut(&mut x, frame, state);
        for i in 0..state {
            dst[i] += pos[i];
        }
    }

    for layer in &weights.encoder.layers {
        encoder_layer(
            config,
            layer,
            &mut x,
            frames,
            profile,
            pool,
            rust_config,
            scratch,
        );
    }

    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut x,
        frames,
        state,
        &weights.encoder.ln_w,
        &weights.encoder.ln_b,
        1e-5,
    );
    profile.encoder_layer_norm += start.elapsed();

    Ok(EncodedAudio {
        values: x,
        frames,
        state,
    })
}

pub fn decoder_logits(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    encoded: &EncodedAudio,
    token_ids: &[usize],
    profile: &mut WhisperOperationProfile,
) -> Result<Vec<f32>> {
    let pool = ThreadPool::new(1);
    decoder_logits_with_rust_config(
        config,
        weights,
        encoded,
        token_ids,
        profile,
        &pool,
        &WhisperRustConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decoder_logits_with_rust_config(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    encoded: &EncodedAudio,
    token_ids: &[usize],
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Result<Vec<f32>> {
    let mut scratch = WhisperForwardScratch::default();
    decoder_logits_with_scratch(
        config,
        weights,
        encoded,
        token_ids,
        profile,
        pool,
        rust_config,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decoder_logits_with_rust_config_and_quantized_logits(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    quantized_output_projection: Option<&QuantizedRows>,
    encoded: &EncodedAudio,
    token_ids: &[usize],
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Result<Vec<f32>> {
    let mut scratch = WhisperForwardScratch::default();
    decoder_logits_with_scratch_and_quantized_logits(
        config,
        weights,
        quantized_output_projection,
        encoded,
        token_ids,
        profile,
        pool,
        rust_config,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decoder_logits_with_scratch(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    encoded: &EncodedAudio,
    token_ids: &[usize],
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperForwardScratch,
) -> Result<Vec<f32>> {
    decoder_logits_with_scratch_and_quantized_logits(
        config,
        weights,
        None,
        encoded,
        token_ids,
        profile,
        pool,
        rust_config,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decoder_logits_with_scratch_and_quantized_logits(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    quantized_output_projection: Option<&QuantizedRows>,
    encoded: &EncodedAudio,
    token_ids: &[usize],
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperForwardScratch,
) -> Result<Vec<f32>> {
    rust_config.validate()?;
    validate_decoder_input(config, encoded, token_ids)?;
    let t = token_ids.len();
    let state = config.n_text_state;
    let mut x = vec![0.0f32; t * state];
    for (pos, token_id) in token_ids.iter().copied().enumerate() {
        let token = row(&weights.decoder.token_embedding, token_id, state);
        let pos_emb = row(&weights.decoder.positional_embedding, pos, state);
        let dst = row_mut(&mut x, pos, state);
        for i in 0..state {
            dst[i] = token[i] + pos_emb[i];
        }
    }

    for layer in &weights.decoder.layers {
        decoder_layer(
            config,
            layer,
            encoded,
            &mut x,
            t,
            profile,
            pool,
            rust_config,
            scratch,
        );
    }

    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut x,
        t,
        state,
        &weights.decoder.ln_w,
        &weights.decoder.ln_b,
        1e-5,
    );
    profile.decoder_layer_norm += start.elapsed();

    let start = std::time::Instant::now();
    let hidden = row(&x, t - 1, state);
    let projection = weights
        .output_projection
        .as_ref()
        .unwrap_or(&weights.decoder.token_embedding);
    logits_from_hidden_into(
        hidden,
        projection,
        quantized_output_projection,
        config.n_vocab,
        state,
        pool,
        rust_config,
        &mut scratch.logits,
    );
    profile.final_logits += start.elapsed();
    Ok(std::mem::take(&mut scratch.logits))
}

pub fn generate_greedy(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    encoded: &EncodedAudio,
    prompt: &[usize],
    generation: &TextGenerationConfig,
    timestamp_begin: Option<usize>,
    profile: &mut WhisperOperationProfile,
) -> Result<Vec<usize>> {
    let pool = ThreadPool::new(1);
    generate_greedy_with_rust_config(
        config,
        weights,
        encoded,
        prompt,
        generation,
        timestamp_begin,
        profile,
        &pool,
        &WhisperRustConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_greedy_with_rust_config(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    encoded: &EncodedAudio,
    prompt: &[usize],
    generation: &TextGenerationConfig,
    timestamp_begin: Option<usize>,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Result<Vec<usize>> {
    generate_greedy_with_rust_config_and_quantized_logits(
        config,
        weights,
        None,
        encoded,
        prompt,
        generation,
        timestamp_begin,
        profile,
        pool,
        rust_config,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_greedy_with_rust_config_and_quantized_logits(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    quantized_output_projection: Option<&QuantizedRows>,
    encoded: &EncodedAudio,
    prompt: &[usize],
    generation: &TextGenerationConfig,
    timestamp_begin: Option<usize>,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Result<Vec<usize>> {
    generate_greedy_with_rust_config_and_quantized_logits_callback::<_, WhisperError>(
        config,
        weights,
        quantized_output_projection,
        encoded,
        prompt,
        generation,
        timestamp_begin,
        profile,
        pool,
        rust_config,
        |_| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_greedy_with_rust_config_and_quantized_logits_callback<F, E>(
    config: &WhisperConfig,
    weights: &WhisperWeights,
    quantized_output_projection: Option<&QuantizedRows>,
    encoded: &EncodedAudio,
    prompt: &[usize],
    generation: &TextGenerationConfig,
    timestamp_begin: Option<usize>,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    on_token: F,
) -> std::result::Result<Vec<usize>, E>
where
    F: FnMut(usize) -> std::result::Result<(), E>,
    E: From<WhisperError>,
{
    let mut decoder = WhisperConditionalDecoder::new_with_rust_config(
        config,
        weights,
        quantized_output_projection,
        generation,
        timestamp_begin,
        profile,
        pool,
        rust_config,
    )?;
    autoregressive::generate_conditional::<_, _, E>(
        &mut decoder,
        encoded,
        prompt,
        generation.max_new_tokens,
        on_token,
    )
}

pub struct WhisperConditionalDecoder<'a> {
    config: &'a WhisperConfig,
    weights: &'a WhisperWeights,
    generation: &'a TextGenerationConfig,
    timestamp_begin: Option<usize>,
    profile: &'a mut WhisperOperationProfile,
    pool: Option<&'a ThreadPool>,
    rust_config: Option<&'a WhisperRustConfig>,
    quantized_output_projection: Option<&'a QuantizedRows>,
    scratch: WhisperForwardScratch,
    sampler: LogitsSampler,
}

impl<'a> WhisperConditionalDecoder<'a> {
    #[allow(dead_code)]
    pub fn new(
        config: &'a WhisperConfig,
        weights: &'a WhisperWeights,
        generation: &'a TextGenerationConfig,
        timestamp_begin: Option<usize>,
        profile: &'a mut WhisperOperationProfile,
    ) -> Result<Self> {
        generation
            .validate()
            .map_err(|err| WhisperError::InvalidInput(err.to_string()))?;
        Ok(Self {
            config,
            weights,
            generation,
            timestamp_begin,
            profile,
            pool: None,
            rust_config: None,
            quantized_output_projection: None,
            scratch: WhisperForwardScratch::default(),
            sampler: LogitsSampler::new(generation.seed),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_rust_config(
        config: &'a WhisperConfig,
        weights: &'a WhisperWeights,
        quantized_output_projection: Option<&'a QuantizedRows>,
        generation: &'a TextGenerationConfig,
        timestamp_begin: Option<usize>,
        profile: &'a mut WhisperOperationProfile,
        pool: &'a ThreadPool,
        rust_config: &'a WhisperRustConfig,
    ) -> Result<Self> {
        generation
            .validate()
            .map_err(|err| WhisperError::InvalidInput(err.to_string()))?;
        rust_config.validate()?;
        Ok(Self {
            config,
            weights,
            generation,
            timestamp_begin,
            profile,
            pool: Some(pool),
            rust_config: Some(rust_config),
            quantized_output_projection,
            scratch: WhisperForwardScratch::default(),
            sampler: LogitsSampler::new(generation.seed),
        })
    }

    fn suppress_timestamps(&self, logits: &mut [f32]) {
        if let Some(timestamp_begin) = self.timestamp_begin {
            for logit in logits.iter_mut().skip(timestamp_begin) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

impl ConditionalAutoregressiveDecoder for WhisperConditionalDecoder<'_> {
    type Condition = EncodedAudio;
    type Logits = Vec<f32>;
    type Error = WhisperError;

    fn max_context_len(&self) -> usize {
        self.config.n_text_ctx
    }

    fn prefill(
        &mut self,
        condition: &Self::Condition,
        input_ids: &[usize],
    ) -> Result<Self::Logits> {
        self.forward(condition, input_ids)
    }

    fn forward(
        &mut self,
        condition: &Self::Condition,
        input_ids: &[usize],
    ) -> Result<Self::Logits> {
        if let (Some(pool), Some(rust_config)) = (self.pool, self.rust_config) {
            decoder_logits_with_scratch_and_quantized_logits(
                self.config,
                self.weights,
                self.quantized_output_projection,
                condition,
                input_ids,
                self.profile,
                pool,
                rust_config,
                &mut self.scratch,
            )
        } else {
            decoder_logits(
                self.config,
                self.weights,
                condition,
                input_ids,
                self.profile,
            )
        }
    }

    fn select_next_token(&mut self, logits: &Self::Logits, history: &[usize]) -> Result<usize> {
        let mut logits = logits.clone();
        self.suppress_timestamps(&mut logits);
        self.sampler
            .select_next_token(&logits, history, self.generation)
            .map_err(|err| WhisperError::InvalidInput(err.to_string()))
    }

    fn should_stop(&self, token_id: usize) -> bool {
        self.generation.eos_token_id == Some(token_id)
    }
}

fn validate_decoder_input(
    config: &WhisperConfig,
    encoded: &EncodedAudio,
    token_ids: &[usize],
) -> Result<()> {
    if encoded.frames != config.n_audio_ctx || encoded.state != config.n_audio_state {
        return Err(WhisperError::InvalidInput(format!(
            "encoded audio shape {}x{} does not match expected {}x{}",
            encoded.frames, encoded.state, config.n_audio_ctx, config.n_audio_state
        )));
    }
    if token_ids.is_empty() {
        return Err(WhisperError::InvalidInput(
            "decoder token_ids must not be empty".to_string(),
        ));
    }
    if token_ids.len() > config.n_text_ctx {
        return Err(WhisperError::InvalidInput(format!(
            "decoder token length {} exceeds text context {}",
            token_ids.len(),
            config.n_text_ctx
        )));
    }
    for (i, token_id) in token_ids.iter().copied().enumerate() {
        if token_id >= config.n_vocab {
            return Err(WhisperError::InvalidInput(format!(
                "token id {token_id} at index {i} exceeds vocab {}",
                config.n_vocab
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encoder_layer(
    config: &WhisperConfig,
    layer: &WhisperEncoderLayerWeights,
    x: &mut [f32],
    frames: usize,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperForwardScratch,
) {
    let state = config.n_audio_state;
    scratch.norm.clear();
    scratch.norm.extend_from_slice(x);
    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut scratch.norm,
        frames,
        state,
        &layer.self_attn_ln_w,
        &layer.self_attn_ln_b,
        1e-5,
    );
    profile.encoder_layer_norm += start.elapsed();

    let start = std::time::Instant::now();
    self_attention_into(
        &scratch.norm,
        frames,
        state,
        config.n_audio_head,
        &layer.self_attn,
        false,
        pool,
        rust_config,
        &mut scratch.attention,
        &mut scratch.attention_out,
    );
    profile.encoder_attention += start.elapsed();
    add_in_place(x, &scratch.attention_out);

    scratch.norm.clear();
    scratch.norm.extend_from_slice(x);
    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut scratch.norm,
        frames,
        state,
        &layer.final_ln_w,
        &layer.final_ln_b,
        1e-5,
    );
    profile.encoder_layer_norm += start.elapsed();

    let start = std::time::Instant::now();
    linear_into(
        &scratch.norm,
        frames,
        state,
        &layer.fc1_w,
        &layer.fc1_b,
        config.n_audio_mlp,
        pool,
        rust_config,
        &mut scratch.hidden,
    );
    gelu_in_place(&mut scratch.hidden);
    linear_into(
        &scratch.hidden,
        frames,
        config.n_audio_mlp,
        &layer.fc2_w,
        &layer.fc2_b,
        state,
        pool,
        rust_config,
        &mut scratch.mlp,
    );
    profile.encoder_mlp += start.elapsed();
    add_in_place(x, &scratch.mlp);
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer(
    config: &WhisperConfig,
    layer: &WhisperDecoderLayerWeights,
    encoded: &EncodedAudio,
    x: &mut [f32],
    tokens: usize,
    profile: &mut WhisperOperationProfile,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperForwardScratch,
) {
    let state = config.n_text_state;
    scratch.norm.clear();
    scratch.norm.extend_from_slice(x);
    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut scratch.norm,
        tokens,
        state,
        &layer.self_attn_ln_w,
        &layer.self_attn_ln_b,
        1e-5,
    );
    profile.decoder_layer_norm += start.elapsed();

    let start = std::time::Instant::now();
    self_attention_into(
        &scratch.norm,
        tokens,
        state,
        config.n_text_head,
        &layer.self_attn,
        true,
        pool,
        rust_config,
        &mut scratch.attention,
        &mut scratch.attention_out,
    );
    profile.decoder_self_attention += start.elapsed();
    add_in_place(x, &scratch.attention_out);

    scratch.norm.clear();
    scratch.norm.extend_from_slice(x);
    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut scratch.norm,
        tokens,
        state,
        &layer.cross_attn_ln_w,
        &layer.cross_attn_ln_b,
        1e-5,
    );
    profile.decoder_layer_norm += start.elapsed();

    let start = std::time::Instant::now();
    cross_attention_into(
        &scratch.norm,
        tokens,
        &encoded.values,
        encoded.frames,
        state,
        config.n_text_head,
        &layer.cross_attn,
        pool,
        rust_config,
        &mut scratch.attention,
        &mut scratch.attention_out,
    );
    profile.decoder_cross_attention += start.elapsed();
    add_in_place(x, &scratch.attention_out);

    scratch.norm.clear();
    scratch.norm.extend_from_slice(x);
    let start = std::time::Instant::now();
    layer_norm_in_place(
        &mut scratch.norm,
        tokens,
        state,
        &layer.final_ln_w,
        &layer.final_ln_b,
        1e-5,
    );
    profile.decoder_layer_norm += start.elapsed();

    let start = std::time::Instant::now();
    linear_into(
        &scratch.norm,
        tokens,
        state,
        &layer.fc1_w,
        &layer.fc1_b,
        config.n_text_mlp,
        pool,
        rust_config,
        &mut scratch.hidden,
    );
    gelu_in_place(&mut scratch.hidden);
    linear_into(
        &scratch.hidden,
        tokens,
        config.n_text_mlp,
        &layer.fc2_w,
        &layer.fc2_b,
        state,
        pool,
        rust_config,
        &mut scratch.mlp,
    );
    profile.decoder_mlp += start.elapsed();
    add_in_place(x, &scratch.mlp);
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn self_attention(
    x: &[f32],
    rows: usize,
    state: usize,
    heads: usize,
    weights: &WhisperAttentionWeights,
    causal: bool,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let mut scratch = WhisperAttentionScratch::default();
    let mut out = Vec::new();
    self_attention_into(
        x,
        rows,
        state,
        heads,
        weights,
        causal,
        pool,
        rust_config,
        &mut scratch,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn self_attention_into(
    x: &[f32],
    rows: usize,
    state: usize,
    heads: usize,
    weights: &WhisperAttentionWeights,
    causal: bool,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperAttentionScratch,
    out: &mut Vec<f32>,
) {
    linear_into(
        x,
        rows,
        state,
        &weights.q_w,
        &weights.q_b,
        state,
        pool,
        rust_config,
        &mut scratch.q,
    );
    linear_optional_bias_into(
        x,
        rows,
        state,
        &weights.k_w,
        weights.k_b.as_deref(),
        state,
        pool,
        rust_config,
        &mut scratch.k,
    );
    linear_into(
        x,
        rows,
        state,
        &weights.v_w,
        &weights.v_b,
        state,
        pool,
        rust_config,
        &mut scratch.v,
    );
    attention_values_into(
        &scratch.q,
        &scratch.k,
        &scratch.v,
        AttentionShape {
            query_rows: rows,
            key_rows: rows,
            state,
            heads,
            causal,
        },
        pool,
        rust_config,
        &mut scratch.scores,
        &mut scratch.values,
    );
    linear_into(
        &scratch.values,
        rows,
        state,
        &weights.out_w,
        &weights.out_b,
        state,
        pool,
        rust_config,
        out,
    );
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn cross_attention(
    query_x: &[f32],
    query_rows: usize,
    key_value_x: &[f32],
    key_value_rows: usize,
    state: usize,
    heads: usize,
    weights: &WhisperAttentionWeights,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let mut scratch = WhisperAttentionScratch::default();
    let mut out = Vec::new();
    cross_attention_into(
        query_x,
        query_rows,
        key_value_x,
        key_value_rows,
        state,
        heads,
        weights,
        pool,
        rust_config,
        &mut scratch,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn cross_attention_into(
    query_x: &[f32],
    query_rows: usize,
    key_value_x: &[f32],
    key_value_rows: usize,
    state: usize,
    heads: usize,
    weights: &WhisperAttentionWeights,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scratch: &mut WhisperAttentionScratch,
    out: &mut Vec<f32>,
) {
    linear_into(
        query_x,
        query_rows,
        state,
        &weights.q_w,
        &weights.q_b,
        state,
        pool,
        rust_config,
        &mut scratch.q,
    );
    linear_optional_bias_into(
        key_value_x,
        key_value_rows,
        state,
        &weights.k_w,
        weights.k_b.as_deref(),
        state,
        pool,
        rust_config,
        &mut scratch.k,
    );
    linear_into(
        key_value_x,
        key_value_rows,
        state,
        &weights.v_w,
        &weights.v_b,
        state,
        pool,
        rust_config,
        &mut scratch.v,
    );
    attention_values_into(
        &scratch.q,
        &scratch.k,
        &scratch.v,
        AttentionShape {
            query_rows,
            key_rows: key_value_rows,
            state,
            heads,
            causal: false,
        },
        pool,
        rust_config,
        &mut scratch.scores,
        &mut scratch.values,
    );
    linear_into(
        &scratch.values,
        query_rows,
        state,
        &weights.out_w,
        &weights.out_b,
        state,
        pool,
        rust_config,
        out,
    );
}

#[allow(dead_code)]
fn attention_values(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let mut scores = Vec::new();
    let mut out = Vec::new();
    attention_values_into(q, k, v, shape, pool, rust_config, &mut scores, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn attention_values_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    scores: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    let AttentionShape {
        query_rows,
        key_rows,
        state,
        heads,
        causal,
    } = shape;
    let head_dim = state / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    out.clear();
    out.resize(query_rows * state, 0.0);
    let work_items = query_rows * key_rows * heads * head_dim;
    if pool.threads() > 1 && work_items >= rust_config.attention_head_parallel_threshold {
        let head_chunks = pool.scoped_parallel_chunks(
            heads,
            rust_config.attention_head_chunk_size,
            |start, end| {
                let mut outputs = Vec::with_capacity(end - start);
                for head in start..end {
                    let head_start = head * head_dim;
                    let mut head_out = vec![0.0f32; query_rows * head_dim];
                    let mut scores = vec![0.0f32; key_rows];
                    for query in 0..query_rows {
                        let max_key = if causal { query + 1 } else { key_rows };
                        for (key, score_slot) in scores.iter_mut().enumerate().take(max_key) {
                            let mut score = 0.0f32;
                            let q_base = query * state + head_start;
                            let k_base = key * state + head_start;
                            for i in 0..head_dim {
                                score += q[q_base + i] * k[k_base + i];
                            }
                            *score_slot = score * scale;
                        }
                        softmax_in_place(&mut scores[..max_key]);
                        let out_base = query * head_dim;
                        for i in 0..head_dim {
                            let mut value = 0.0f32;
                            for key in 0..max_key {
                                value += scores[key] * v[key * state + head_start + i];
                            }
                            head_out[out_base + i] = value;
                        }
                    }
                    outputs.push((head, head_out));
                }
                outputs
            },
        );
        for (head, head_out) in head_chunks.into_iter().flatten() {
            let head_start = head * head_dim;
            for query in 0..query_rows {
                let src = row(&head_out, query, head_dim);
                let dst_start = query * state + head_start;
                out[dst_start..dst_start + head_dim].copy_from_slice(src);
            }
        }
        return;
    }

    scores.clear();
    scores.resize(key_rows, 0.0);
    for head in 0..heads {
        let head_start = head * head_dim;
        for query in 0..query_rows {
            let max_key = if causal { query + 1 } else { key_rows };
            for (key, score_slot) in scores.iter_mut().enumerate().take(max_key) {
                let mut score = 0.0f32;
                let q_base = query * state + head_start;
                let k_base = key * state + head_start;
                for i in 0..head_dim {
                    score += q[q_base + i] * k[k_base + i];
                }
                *score_slot = score * scale;
            }
            softmax_in_place(&mut scores[..max_key]);
            let out_base = query * state + head_start;
            for i in 0..head_dim {
                let mut value = 0.0f32;
                for key in 0..max_key {
                    value += scores[key] * v[key * state + head_start + i];
                }
                out[out_base + i] = value;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn linear(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    out_features: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let mut out = Vec::new();
    linear_into(
        x,
        rows,
        in_features,
        weight,
        bias,
        out_features,
        pool,
        rust_config,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn linear_into(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    out_features: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    out: &mut Vec<f32>,
) {
    linear_optional_bias_into(
        x,
        rows,
        in_features,
        weight,
        Some(bias),
        out_features,
        pool,
        rust_config,
        out,
    );
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn linear_optional_bias(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let mut out = Vec::new();
    linear_optional_bias_into(
        x,
        rows,
        in_features,
        weight,
        bias,
        out_features,
        pool,
        rust_config,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn linear_optional_bias_into(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    out: &mut Vec<f32>,
) {
    let shape = DenseShape::new(rows, in_features, out_features);
    if pool.threads() > 1 && shape.work_items() >= rust_config.dense_parallel_threshold {
        let chunks = pool.scoped_parallel_chunks(
            shape.out_len(),
            rust_config.dense_chunk_size,
            |start, end| {
                let mut values = Vec::with_capacity(end - start);
                for index in start..end {
                    let r = index / out_features;
                    let o = index % out_features;
                    let b = bias.map_or(0.0, |bias| bias[o]);
                    values.push(b + dot(row(x, r, in_features), row(weight, o, in_features)));
                }
                values
            },
        );

        out.clear();
        out.reserve(shape.out_len());
        for mut chunk in chunks {
            out.append(&mut chunk);
        }
        return;
    }

    out.clear();
    out.resize(rows * out_features, 0.0);
    for r in 0..rows {
        let src = row(x, r, in_features);
        let dst = row_mut(out, r, out_features);
        for o in 0..out_features {
            let b = bias.map_or(0.0, |bias| bias[o]);
            dst[o] = b + dot(src, row(weight, o, in_features));
        }
    }
}

#[allow(dead_code)]
fn logits_from_hidden(
    hidden: &[f32],
    projection: &[f32],
    vocab_size: usize,
    state: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let mut logits = Vec::new();
    logits_from_hidden_into(
        hidden,
        projection,
        None,
        vocab_size,
        state,
        pool,
        rust_config,
        &mut logits,
    );
    logits
}

#[allow(clippy::too_many_arguments)]
fn logits_from_hidden_into(
    hidden: &[f32],
    projection: &[f32],
    quantized_projection: Option<&QuantizedRows>,
    vocab_size: usize,
    state: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
    logits: &mut Vec<f32>,
) {
    if pool.threads() > 1 {
        let chunks =
            pool.scoped_parallel_chunks(vocab_size, rust_config.logits_chunk_size, |start, end| {
                let mut values = Vec::with_capacity(end - start);
                for token_id in start..end {
                    let logit = quantized_projection.map_or_else(
                        || dot(hidden, row(projection, token_id, state)),
                        |projection| quantized_dot(hidden, projection, token_id),
                    );
                    values.push(logit);
                }
                values
            });
        logits.clear();
        logits.reserve(vocab_size);
        for mut chunk in chunks {
            logits.append(&mut chunk);
        }
        return;
    }

    logits.clear();
    logits.resize(vocab_size, 0.0);
    for (token_id, logit) in logits.iter_mut().enumerate() {
        *logit = quantized_projection.map_or_else(
            || dot(hidden, row(projection, token_id, state)),
            |projection| quantized_dot(hidden, projection, token_id),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d_features(
    x: &[f32],
    in_channels: usize,
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let out_frames = conv_out_frames(frames, stride);
    let work_items = out_frames * out_channels * in_channels * 3;
    if pool.threads() > 1 && work_items >= rust_config.dense_parallel_threshold {
        let chunks = pool.scoped_parallel_chunks(
            out_frames * out_channels,
            rust_config.dense_chunk_size,
            |start, end| {
                let mut values = Vec::with_capacity(end - start);
                for index in start..end {
                    let t = index / out_channels;
                    let o = index % out_channels;
                    let mut sum = bias[o];
                    for i in 0..in_channels {
                        for k in 0..3 {
                            let Some(src_t) = conv_source_index(t, stride, k, frames) else {
                                continue;
                            };
                            sum += x[i * frames + src_t] * weight[(o * in_channels + i) * 3 + k];
                        }
                    }
                    values.push(sum);
                }
                values
            },
        );
        let mut out = Vec::with_capacity(out_frames * out_channels);
        for mut chunk in chunks {
            out.append(&mut chunk);
        }
        return out;
    }

    let mut out = vec![0.0f32; out_frames * out_channels];
    for t in 0..out_frames {
        for o in 0..out_channels {
            let mut sum = bias[o];
            for i in 0..in_channels {
                for k in 0..3 {
                    let Some(src_t) = conv_source_index(t, stride, k, frames) else {
                        continue;
                    };
                    sum += x[i * frames + src_t] * weight[(o * in_channels + i) * 3 + k];
                }
            }
            out[t * out_channels + o] = sum;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn conv1d_frame_major(
    x: &[f32],
    frames: usize,
    in_channels: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
    pool: &ThreadPool,
    rust_config: &WhisperRustConfig,
) -> Vec<f32> {
    let out_frames = conv_out_frames(frames, stride);
    let work_items = out_frames * out_channels * in_channels * 3;
    if pool.threads() > 1 && work_items >= rust_config.dense_parallel_threshold {
        let chunks = pool.scoped_parallel_chunks(
            out_frames * out_channels,
            rust_config.dense_chunk_size,
            |start, end| {
                let mut values = Vec::with_capacity(end - start);
                for index in start..end {
                    let t = index / out_channels;
                    let o = index % out_channels;
                    let mut sum = bias[o];
                    for i in 0..in_channels {
                        for k in 0..3 {
                            let Some(src_t) = conv_source_index(t, stride, k, frames) else {
                                continue;
                            };
                            sum +=
                                x[src_t * in_channels + i] * weight[(o * in_channels + i) * 3 + k];
                        }
                    }
                    values.push(sum);
                }
                values
            },
        );
        let mut out = Vec::with_capacity(out_frames * out_channels);
        for mut chunk in chunks {
            out.append(&mut chunk);
        }
        return out;
    }

    let mut out = vec![0.0f32; out_frames * out_channels];
    for t in 0..out_frames {
        for o in 0..out_channels {
            let mut sum = bias[o];
            for i in 0..in_channels {
                for k in 0..3 {
                    let Some(src_t) = conv_source_index(t, stride, k, frames) else {
                        continue;
                    };
                    sum += x[src_t * in_channels + i] * weight[(o * in_channels + i) * 3 + k];
                }
            }
            out[t * out_channels + o] = sum;
        }
    }
    out
}

fn conv_out_frames(frames: usize, stride: usize) -> usize {
    frames.div_ceil(stride)
}

fn conv_source_index(out_t: usize, stride: usize, kernel: usize, frames: usize) -> Option<usize> {
    let center = out_t * stride + kernel;
    if center == 0 {
        return None;
    }
    let src = center - 1;
    (src < frames).then_some(src)
}

fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src.iter()) {
        *dst += src;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::whisper::weights::{
        WhisperAttentionWeights, WhisperDecoderLayerWeights, WhisperDecoderWeights,
        WhisperEncoderLayerWeights, WhisperEncoderWeights, WhisperWeights, WhisperWeightsManifest,
    };

    #[test]
    fn synthetic_encoder_returns_configured_audio_shape() -> Result<()> {
        let (config, preprocessor, weights) = tiny_synthetic_model();
        let features = LogMelSpectrogram {
            values: vec![0.0; config.n_mels * preprocessor.n_frames],
            n_mels: config.n_mels,
            n_frames: preprocessor.n_frames,
        };
        let mut profile = WhisperOperationProfile::default();

        let encoded = encode_audio(&config, &preprocessor, &weights, &features, &mut profile)?;

        assert_eq!(encoded.frames, config.n_audio_ctx);
        assert_eq!(encoded.state, config.n_audio_state);
        assert_eq!(
            encoded.values.len(),
            config.n_audio_ctx * config.n_audio_state
        );
        Ok(())
    }

    #[test]
    fn synthetic_decoder_returns_vocab_logits() -> Result<()> {
        let (config, _, weights) = tiny_synthetic_model();
        let encoded = EncodedAudio {
            values: vec![0.0; config.n_audio_ctx * config.n_audio_state],
            frames: config.n_audio_ctx,
            state: config.n_audio_state,
        };
        let mut profile = WhisperOperationProfile::default();

        let logits = decoder_logits(&config, &weights, &encoded, &[0, 1], &mut profile)?;

        assert_eq!(logits.len(), config.n_vocab);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn conditional_decoder_uses_encoded_audio_condition() -> Result<()> {
        let (config, _, weights) = tiny_synthetic_model();
        let encoded = EncodedAudio {
            values: vec![0.0; config.n_audio_ctx * config.n_audio_state],
            frames: config.n_audio_ctx,
            state: config.n_audio_state,
        };
        let generation = TextGenerationConfig::new(2).with_eos_token_id(Some(2));
        let mut profile = WhisperOperationProfile::default();
        let mut decoder =
            WhisperConditionalDecoder::new(&config, &weights, &generation, None, &mut profile)?;

        let output = autoregressive::generate_conditional::<_, _, WhisperError>(
            &mut decoder,
            &encoded,
            &[0],
            2,
            |_| Ok(()),
        )?;

        assert!(!output.is_empty());
        assert!(output.len() <= 3);
        Ok(())
    }

    #[test]
    fn decoder_kv_cache_exposes_shared_cache_contract() {
        let (config, _, _) = tiny_synthetic_model();
        let mut cache = WhisperDecoderKvCache::new(&config);

        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.max_seq_len(), config.n_text_ctx);
        assert_eq!(cache.layers.len(), config.n_text_layer);
        cache.seq_len = 2;
        cache.layers[0].self_keys[0] = 1.0;

        cache.clear();

        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.layers[0].self_keys[0], 0.0);
    }

    #[test]
    fn threaded_encoder_and_decoder_match_single_threaded() -> Result<()> {
        let (config, preprocessor, weights) = tiny_synthetic_model();
        let features = LogMelSpectrogram {
            values: (0..config.n_mels * preprocessor.n_frames)
                .map(|i| (i % 11) as f32 * 0.01)
                .collect(),
            n_mels: config.n_mels,
            n_frames: preprocessor.n_frames,
        };
        let mut single_profile = WhisperOperationProfile::default();
        let single_encoded = encode_audio(
            &config,
            &preprocessor,
            &weights,
            &features,
            &mut single_profile,
        )?;

        let pool = ThreadPool::new(3);
        let rust_config = WhisperRustConfig {
            threads: 3,
            dense_parallel_threshold: 1,
            dense_chunk_size: 2,
            logits_chunk_size: 2,
            attention_head_parallel_threshold: 1,
            attention_head_chunk_size: 1,
            quantized_weights: false,
        };
        let mut threaded_profile = WhisperOperationProfile::default();
        let threaded_encoded = encode_audio_with_rust_config(
            &config,
            &preprocessor,
            &weights,
            &features,
            &mut threaded_profile,
            &pool,
            &rust_config,
        )?;

        assert_eq!(threaded_encoded, single_encoded);

        let mut single_profile = WhisperOperationProfile::default();
        let single_logits = decoder_logits(
            &config,
            &weights,
            &single_encoded,
            &[0, 1],
            &mut single_profile,
        )?;
        let mut threaded_profile = WhisperOperationProfile::default();
        let threaded_logits = decoder_logits_with_rust_config(
            &config,
            &weights,
            &threaded_encoded,
            &[0, 1],
            &mut threaded_profile,
            &pool,
            &rust_config,
        )?;

        assert_eq!(threaded_logits, single_logits);
        Ok(())
    }

    #[test]
    fn quantized_logits_path_returns_finite_values() -> Result<()> {
        let (config, _, weights) = tiny_synthetic_model();
        let encoded = EncodedAudio {
            values: vec![0.0; config.n_audio_ctx * config.n_audio_state],
            frames: config.n_audio_ctx,
            state: config.n_audio_state,
        };
        let quantized_projection = QuantizedRows::from_f32(
            &weights.decoder.token_embedding,
            config.n_vocab,
            config.n_text_state,
        );
        let pool = ThreadPool::new(1);
        let rust_config = WhisperRustConfig::default().with_quantized_weights(true);
        let mut profile = WhisperOperationProfile::default();

        let logits = decoder_logits_with_rust_config_and_quantized_logits(
            &config,
            &weights,
            Some(&quantized_projection),
            &encoded,
            &[0, 1],
            &mut profile,
            &pool,
            &rust_config,
        )?;

        assert_eq!(logits.len(), config.n_vocab);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    fn tiny_synthetic_model() -> (WhisperConfig, WhisperPreprocessorConfig, WhisperWeights) {
        let config = WhisperConfig {
            n_mels: 2,
            n_audio_ctx: 50,
            n_audio_state: 2,
            n_audio_head: 1,
            n_audio_layer: 1,
            n_audio_mlp: 4,
            n_vocab: 3,
            n_text_ctx: 4,
            n_text_state: 2,
            n_text_head: 1,
            n_text_layer: 1,
            n_text_mlp: 4,
        };
        let preprocessor = WhisperPreprocessorConfig {
            sample_rate: 16_000,
            chunk_length_seconds: 1,
            n_fft: 400,
            hop_length: 160,
            n_mels: 2,
            n_samples: 16_000,
            n_frames: 100,
            padding_value: 0.0,
            return_attention_mask: false,
        };
        let attn = WhisperAttentionWeights {
            q_w: zeros(4),
            q_b: zeros(2),
            k_w: zeros(4),
            k_b: None,
            v_w: zeros(4),
            v_b: zeros(2),
            out_w: zeros(4),
            out_b: zeros(2),
        };
        let encoder_layer = WhisperEncoderLayerWeights {
            self_attn: attn.clone(),
            self_attn_ln_w: ones(2),
            self_attn_ln_b: zeros(2),
            fc1_w: zeros(8),
            fc1_b: zeros(4),
            fc2_w: zeros(8),
            fc2_b: zeros(2),
            final_ln_w: ones(2),
            final_ln_b: zeros(2),
        };
        let decoder_layer = WhisperDecoderLayerWeights {
            self_attn: attn.clone(),
            self_attn_ln_w: ones(2),
            self_attn_ln_b: zeros(2),
            cross_attn: attn,
            cross_attn_ln_w: ones(2),
            cross_attn_ln_b: zeros(2),
            fc1_w: zeros(8),
            fc1_b: zeros(4),
            fc2_w: zeros(8),
            fc2_b: zeros(2),
            final_ln_w: ones(2),
            final_ln_b: zeros(2),
        };
        let weights = WhisperWeights {
            manifest: WhisperWeightsManifest {
                tensor_count: 0,
                encoder_layers: 1,
                decoder_layers: 1,
                tied_output_projection: true,
            },
            encoder: WhisperEncoderWeights {
                conv1_w: zeros(12),
                conv1_b: zeros(2),
                conv2_w: zeros(12),
                conv2_b: zeros(2),
                positional_embedding: zeros(100),
                layers: vec![encoder_layer],
                ln_w: ones(2),
                ln_b: zeros(2),
            },
            decoder: WhisperDecoderWeights {
                token_embedding: vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1],
                positional_embedding: zeros(8),
                layers: vec![decoder_layer],
                ln_w: ones(2),
                ln_b: zeros(2),
            },
            output_projection: None,
        };
        (config, preprocessor, weights)
    }

    fn zeros(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn ones(len: usize) -> Vec<f32> {
        vec![1.0; len]
    }
}
