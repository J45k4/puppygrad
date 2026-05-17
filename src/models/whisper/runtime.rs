use std::path::Path;

use crate::models::cpu::QuantizedRows;
use crate::runtime::thread_pool::ThreadPool;

use super::{
    decoder_logits_with_rust_config_and_quantized_logits, encode_audio_with_rust_config,
    generate_greedy_with_rust_config_and_quantized_logits,
    generate_greedy_with_rust_config_and_quantized_logits_callback, load_whisper_config,
    load_whisper_preprocessor_config, load_whisper_weights, prepare_whisper_assets, EncodedAudio,
    LogMelSpectrogram, Result, WhisperAssetPaths, WhisperBackendConfig, WhisperConfig,
    WhisperError, WhisperOperationProfile, WhisperPreprocessorConfig, WhisperRustConfig,
    WhisperSize, WhisperTokenizer, WhisperWeights,
};

#[derive(Clone)]
pub struct WhisperRuntime {
    pub size: WhisperSize,
    pub paths: WhisperAssetPaths,
    pub config: WhisperConfig,
    pub preprocessor: WhisperPreprocessorConfig,
    pub tokenizer: WhisperTokenizer,
    pub weights: WhisperWeights,
    pub rust_config: WhisperRustConfig,
    quantized_output_projection: Option<QuantizedRows>,
    thread_pool: ThreadPool,
}

impl WhisperRuntime {
    pub fn from_dir(size: WhisperSize, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with_rust_config(size, model_dir, WhisperRustConfig::for_size(size))
    }

    pub fn from_dir_with_backend(
        size: WhisperSize,
        model_dir: impl AsRef<Path>,
        backend: WhisperBackendConfig,
    ) -> Result<Self> {
        match backend {
            WhisperBackendConfig::Rust(rust_config) => {
                Self::from_dir_with_rust_config(size, model_dir, rust_config)
            }
            WhisperBackendConfig::Gpu => Err(WhisperError::InvalidInput(
                "Whisper GPU backend hook is present, but GPU kernels are not implemented yet"
                    .to_string(),
            )),
        }
    }

    pub fn from_dir_with_rust_config(
        size: WhisperSize,
        model_dir: impl AsRef<Path>,
        rust_config: WhisperRustConfig,
    ) -> Result<Self> {
        rust_config.validate()?;
        let paths = WhisperAssetPaths::new(model_dir.as_ref());
        let config = load_whisper_config(&paths.config)?;
        let preprocessor = load_whisper_preprocessor_config(&paths.preprocessor_config)?;
        let tokenizer = WhisperTokenizer::from_file(&paths.tokenizer)?;
        let weights = load_whisper_weights(&paths.weights, &config)?;
        let quantized_output_projection =
            quantized_output_projection(&weights, rust_config.quantized_weights);
        let thread_pool = ThreadPool::new(rust_config.threads);
        Ok(Self {
            size,
            paths,
            config,
            preprocessor,
            tokenizer,
            weights,
            rust_config,
            quantized_output_projection,
            thread_pool,
        })
    }

    pub fn prepare_from_huggingface(
        size: WhisperSize,
        model_id: Option<&str>,
        revision: &str,
        model_dir: impl AsRef<Path>,
        download: bool,
    ) -> Result<Self> {
        Self::prepare_from_huggingface_with_rust_config(
            size,
            model_id,
            revision,
            model_dir,
            download,
            WhisperRustConfig::for_size(size),
        )
    }

    pub fn prepare_from_huggingface_with_backend(
        size: WhisperSize,
        model_id: Option<&str>,
        revision: &str,
        model_dir: impl AsRef<Path>,
        download: bool,
        backend: WhisperBackendConfig,
    ) -> Result<Self> {
        match backend {
            WhisperBackendConfig::Rust(rust_config) => {
                Self::prepare_from_huggingface_with_rust_config(
                    size,
                    model_id,
                    revision,
                    model_dir,
                    download,
                    rust_config,
                )
            }
            WhisperBackendConfig::Gpu => Err(WhisperError::InvalidInput(
                "Whisper GPU backend hook is present, but GPU kernels are not implemented yet"
                    .to_string(),
            )),
        }
    }

    pub fn prepare_from_huggingface_with_rust_config(
        size: WhisperSize,
        model_id: Option<&str>,
        revision: &str,
        model_dir: impl AsRef<Path>,
        download: bool,
        rust_config: WhisperRustConfig,
    ) -> Result<Self> {
        rust_config.validate()?;
        let paths = prepare_whisper_assets(size, model_id, revision, model_dir, download)?;
        let config = load_whisper_config(&paths.config)?;
        let preprocessor = load_whisper_preprocessor_config(&paths.preprocessor_config)?;
        let tokenizer = WhisperTokenizer::from_file(&paths.tokenizer)?;
        let weights = load_whisper_weights(&paths.weights, &config)?;
        let quantized_output_projection =
            quantized_output_projection(&weights, rust_config.quantized_weights);
        let thread_pool = ThreadPool::new(rust_config.threads);
        Ok(Self {
            size,
            paths,
            config,
            preprocessor,
            tokenizer,
            weights,
            rust_config,
            quantized_output_projection,
            thread_pool,
        })
    }

    pub fn metadata_lines(&self) -> Vec<String> {
        vec![
            format!("size: {}", self.size),
            format!("model_id: {}", self.size.model_id()),
            format!("model_dir: {}", self.paths.model_dir.display()),
            format!("english_only: {}", self.size.is_english_only()),
            format!(
                "approx_parameters: {}M",
                self.size.approx_parameters_millions()
            ),
            format!("approx_vram: {}GB", self.size.approx_vram_gb()),
            format!("relative_speed: {}", self.size.relative_speed()),
            format!(
                "config: n_mels={} audio_ctx={} audio_state={} audio_heads={} audio_layers={} text_ctx={} text_state={} text_heads={} text_layers={} vocab={}",
                self.config.n_mels,
                self.config.n_audio_ctx,
                self.config.n_audio_state,
                self.config.n_audio_head,
                self.config.n_audio_layer,
                self.config.n_text_ctx,
                self.config.n_text_state,
                self.config.n_text_head,
                self.config.n_text_layer,
                self.config.n_vocab
            ),
            format!(
                "preprocessor: sample_rate={} chunk_seconds={} n_fft={} hop_length={} n_mels={} n_samples={} n_frames={}",
                self.preprocessor.sample_rate,
                self.preprocessor.chunk_length_seconds,
                self.preprocessor.n_fft,
                self.preprocessor.hop_length,
                self.preprocessor.n_mels,
                self.preprocessor.n_samples,
                self.preprocessor.n_frames
            ),
            format!(
                "weights: tensors={} encoder_layers={} decoder_layers={} tied_output_projection={}",
                self.weights.manifest.tensor_count,
                self.weights.manifest.encoder_layers,
                self.weights.manifest.decoder_layers,
                self.weights.manifest.tied_output_projection
            ),
            format!(
                "rust: threads={} dense_threshold={} dense_chunk={} logits_chunk={} attention_head_threshold={} attention_head_chunk={} logits_weights={}",
                self.rust_config.threads,
                self.rust_config.dense_parallel_threshold,
                self.rust_config.dense_chunk_size,
                self.rust_config.logits_chunk_size,
                self.rust_config.attention_head_parallel_threshold,
                self.rust_config.attention_head_chunk_size,
                if self.rust_config.quantized_weights { "int8" } else { "f32" }
            ),
        ]
    }

    pub fn encode_audio(
        &self,
        features: &LogMelSpectrogram,
        profile: &mut WhisperOperationProfile,
    ) -> Result<EncodedAudio> {
        encode_audio_with_rust_config(
            &self.config,
            &self.preprocessor,
            &self.weights,
            features,
            profile,
            &self.thread_pool,
            &self.rust_config,
        )
    }

    pub fn decoder_logits(
        &self,
        encoded: &EncodedAudio,
        token_ids: &[usize],
        profile: &mut WhisperOperationProfile,
    ) -> Result<Vec<f32>> {
        decoder_logits_with_rust_config_and_quantized_logits(
            &self.config,
            &self.weights,
            self.quantized_output_projection.as_ref(),
            encoded,
            token_ids,
            profile,
            &self.thread_pool,
            &self.rust_config,
        )
    }

    pub fn generate_greedy(
        &self,
        encoded: &EncodedAudio,
        prompt: &[usize],
        generation: &crate::models::generation::TextGenerationConfig,
        suppress_timestamps: bool,
        profile: &mut WhisperOperationProfile,
    ) -> Result<Vec<usize>> {
        let timestamp_begin = suppress_timestamps
            .then_some(self.tokenizer.special_tokens().timestamp_begin)
            .flatten();
        generate_greedy_with_rust_config_and_quantized_logits(
            &self.config,
            &self.weights,
            self.quantized_output_projection.as_ref(),
            encoded,
            prompt,
            generation,
            timestamp_begin,
            profile,
            &self.thread_pool,
            &self.rust_config,
        )
    }

    pub fn stream_greedy_tokens<F, E>(
        &self,
        encoded: &EncodedAudio,
        prompt: &[usize],
        generation: &crate::models::generation::TextGenerationConfig,
        suppress_timestamps: bool,
        profile: &mut WhisperOperationProfile,
        on_token: F,
    ) -> std::result::Result<Vec<usize>, E>
    where
        F: FnMut(usize) -> std::result::Result<(), E>,
        E: From<WhisperError>,
    {
        let timestamp_begin = suppress_timestamps
            .then_some(self.tokenizer.special_tokens().timestamp_begin)
            .flatten();
        generate_greedy_with_rust_config_and_quantized_logits_callback(
            &self.config,
            &self.weights,
            self.quantized_output_projection.as_ref(),
            encoded,
            prompt,
            generation,
            timestamp_begin,
            profile,
            &self.thread_pool,
            &self.rust_config,
            on_token,
        )
    }
}

fn quantized_output_projection(weights: &WhisperWeights, enabled: bool) -> Option<QuantizedRows> {
    enabled.then(|| {
        let projection = weights
            .output_projection
            .as_ref()
            .unwrap_or(&weights.decoder.token_embedding);
        let rows = if weights.output_projection.is_some() {
            weights
                .output_projection
                .as_ref()
                .map(|projection| projection.len() / weights.decoder.ln_w.len())
                .unwrap()
        } else {
            weights.decoder.token_embedding.len() / weights.decoder.ln_w.len()
        };
        QuantizedRows::from_f32(projection, rows, weights.decoder.ln_w.len())
    })
}
