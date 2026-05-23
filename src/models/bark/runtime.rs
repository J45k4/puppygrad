use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::{
    build_coarse_input, build_semantic_input_with_history, decode_encodec_audio,
    generate_coarse_codes_with_history_and_progress, generate_fine_codes_with_history_and_progress,
    generate_semantic_tokens_with_progress, load_bark_causal_transformer_weights,
    load_bark_encodec_decoder_weights, load_bark_fine_transformer_weights,
    load_bark_history_prompt, load_bark_native_weight_manifest, BarkAssetPaths,
    BarkCausalTransformer, BarkConfig, BarkError, BarkFineTransformer, BarkGenerationConfig,
    BarkTokenizer, Result,
};

#[derive(Clone, Debug, PartialEq)]
pub struct BarkRuntimeOptions {
    pub text: String,
    pub voice_preset: Option<String>,
    pub seed: u64,
    pub threads: usize,
    pub semantic_sampling: BarkSamplingOptions,
    pub coarse_sampling: BarkSamplingOptions,
    pub fine_sampling: BarkSamplingOptions,
    pub max_semantic_tokens: Option<usize>,
}

impl BarkRuntimeOptions {
    pub fn from_generation_config(
        text: impl Into<String>,
        voice_preset: Option<String>,
        seed: u64,
        generation_config: &BarkGenerationConfig,
    ) -> Self {
        Self {
            text: text.into(),
            voice_preset,
            seed,
            threads: default_bark_threads(),
            semantic_sampling: BarkSamplingOptions::from_semantic_generation_config(
                generation_config,
            ),
            coarse_sampling: BarkSamplingOptions::from_coarse_generation_config(generation_config),
            fine_sampling: BarkSamplingOptions::from_fine_generation_config(generation_config),
            max_semantic_tokens: Some(generation_config.semantic_config.max_new_tokens),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(BarkError::InvalidInput(
                "text must not be empty".to_string(),
            ));
        }
        self.semantic_sampling.validate("semantic")?;
        self.coarse_sampling.validate("coarse")?;
        self.fine_sampling.validate("fine")?;
        if self.threads == 0 {
            return Err(BarkError::InvalidInput("threads must be > 0".to_string()));
        }
        if matches!(self.max_semantic_tokens, Some(0)) {
            return Err(BarkError::InvalidInput(
                "max_semantic_tokens must be > 0 when set".to_string(),
            ));
        }
        Ok(())
    }
}

fn default_bark_threads() -> usize {
    std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
        .clamp(1, 8)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkSamplingOptions {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl BarkSamplingOptions {
    pub fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
        }
    }

    fn from_semantic_generation_config(config: &BarkGenerationConfig) -> Self {
        Self::new(
            config.semantic_config.temperature,
            Some(config.semantic_config.top_k),
            Some(config.semantic_config.top_p),
        )
    }

    fn from_coarse_generation_config(config: &BarkGenerationConfig) -> Self {
        Self::new(
            config.coarse_acoustics_config.temperature,
            Some(config.coarse_acoustics_config.top_k),
            Some(config.coarse_acoustics_config.top_p),
        )
    }

    fn from_fine_generation_config(config: &BarkGenerationConfig) -> Self {
        Self::new(
            config.fine_acoustics_config.temperature,
            Some(config.fine_acoustics_config.top_k),
            Some(config.fine_acoustics_config.top_p),
        )
    }

    pub fn validate(&self, name: &str) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(BarkError::InvalidInput(format!(
                "{name} temperature must be finite and >= 0"
            )));
        }
        if matches!(self.top_k, Some(0)) {
            return Err(BarkError::InvalidInput(format!(
                "{name} top_k must be > 0 when set"
            )));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(BarkError::InvalidInput(format!(
                    "{name} top_p must be finite and in (0, 1]"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkAudioOutput {
    pub sample_rate: usize,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkGenerationTrace {
    pub audio: BarkAudioOutput,
    pub semantic_tokens: Vec<usize>,
    pub coarse_codes: Vec<Vec<usize>>,
    pub fine_codes: Vec<Vec<usize>>,
    pub profile: BarkRuntimeProfile,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BarkRuntimeProfile {
    pub inspect_time: Duration,
    pub history_prompt_time: Duration,
    pub tokenizer_time: Duration,
    pub semantic_load_time: Duration,
    pub semantic_generation_time: Duration,
    pub coarse_load_time: Duration,
    pub coarse_generation_time: Duration,
    pub fine_load_time: Duration,
    pub fine_generation_time: Duration,
    pub encodec_load_time: Duration,
    pub encodec_decode_time: Duration,
    pub total_time: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarkProgressStage {
    InspectNativeWeights,
    LoadHistoryPrompt,
    LoadTokenizer,
    LoadSemanticWeights,
    GenerateSemanticTokens,
    LoadCoarseWeights,
    GenerateCoarseCodes,
    LoadFineWeights,
    GenerateFineCodes,
    LoadEncodecWeights,
    DecodeEncodecAudio,
}

impl BarkProgressStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::InspectNativeWeights => "inspect native weights",
            Self::LoadHistoryPrompt => "load history prompt",
            Self::LoadTokenizer => "load tokenizer",
            Self::LoadSemanticWeights => "load semantic weights",
            Self::GenerateSemanticTokens => "generate semantic tokens",
            Self::LoadCoarseWeights => "load coarse weights",
            Self::GenerateCoarseCodes => "generate coarse codes",
            Self::LoadFineWeights => "load fine weights",
            Self::GenerateFineCodes => "generate fine codes",
            Self::LoadEncodecWeights => "load EnCodec weights",
            Self::DecodeEncodecAudio => "decode EnCodec audio",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarkProgressStatus {
    Started,
    Advanced,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarkProgressEvent {
    pub stage: BarkProgressStage,
    pub status: BarkProgressStatus,
    pub current: Option<usize>,
    pub total: Option<usize>,
    pub elapsed: Option<Duration>,
}

impl BarkProgressEvent {
    fn started(stage: BarkProgressStage) -> Self {
        Self {
            stage,
            status: BarkProgressStatus::Started,
            current: None,
            total: None,
            elapsed: None,
        }
    }

    fn advanced(stage: BarkProgressStage, current: usize, total: usize) -> Self {
        Self {
            stage,
            status: BarkProgressStatus::Advanced,
            current: Some(current),
            total: Some(total),
            elapsed: None,
        }
    }

    fn finished(stage: BarkProgressStage, elapsed: Duration) -> Self {
        Self {
            stage,
            status: BarkProgressStatus::Finished,
            current: None,
            total: None,
            elapsed: Some(elapsed),
        }
    }
}

impl BarkRuntimeProfile {
    pub fn model_stage_time(&self) -> Duration {
        self.semantic_generation_time
            + self.coarse_generation_time
            + self.fine_generation_time
            + self.encodec_decode_time
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkNativeRuntimeInfo {
    pub model_dir: PathBuf,
    pub native_weight_path: PathBuf,
    pub tensor_count: usize,
    pub f32_tensor_count: usize,
}

pub fn inspect_bark_rust_runtime(paths: &BarkAssetPaths) -> Result<BarkNativeRuntimeInfo> {
    let manifest = load_bark_native_weight_manifest(paths)?;
    Ok(BarkNativeRuntimeInfo {
        model_dir: paths.model_dir.clone(),
        native_weight_path: paths.native_weights.clone(),
        tensor_count: manifest.tensor_count,
        f32_tensor_count: manifest.f32_tensor_count,
    })
}

pub fn generate_bark_rust(
    paths: &BarkAssetPaths,
    config: &BarkConfig,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
) -> Result<BarkAudioOutput> {
    generate_bark_rust_trace(paths, config, generation_config, options).map(|trace| trace.audio)
}

pub fn generate_bark_rust_trace(
    paths: &BarkAssetPaths,
    config: &BarkConfig,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
) -> Result<BarkGenerationTrace> {
    generate_bark_rust_trace_with_progress(paths, config, generation_config, options, |_| {})
}

pub fn generate_bark_rust_trace_with_progress(
    paths: &BarkAssetPaths,
    config: &BarkConfig,
    generation_config: &BarkGenerationConfig,
    options: &BarkRuntimeOptions,
    mut progress: impl FnMut(BarkProgressEvent),
) -> Result<BarkGenerationTrace> {
    let total_start = Instant::now();
    let mut profile = BarkRuntimeProfile::default();
    validate_bark_rust_options(options)?;
    progress(BarkProgressEvent::started(
        BarkProgressStage::InspectNativeWeights,
    ));
    let start = Instant::now();
    let info = inspect_bark_rust_runtime(paths)?;
    profile.inspect_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::InspectNativeWeights,
        profile.inspect_time,
    ));
    progress(BarkProgressEvent::started(
        BarkProgressStage::LoadHistoryPrompt,
    ));
    let start = Instant::now();
    let history_prompt = options
        .voice_preset
        .as_deref()
        .map(|voice_preset| load_bark_history_prompt(paths, voice_preset, generation_config))
        .transpose()?;
    profile.history_prompt_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::LoadHistoryPrompt,
        profile.history_prompt_time,
    ));
    progress(BarkProgressEvent::started(BarkProgressStage::LoadTokenizer));
    let start = Instant::now();
    let tokenizer = BarkTokenizer::from_model_files(
        &paths.vocab,
        &paths.tokenizer_config,
        &paths.special_tokens_map,
        generation_config.semantic_config.max_input_semantic_length,
    )?;
    profile.tokenizer_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::LoadTokenizer,
        profile.tokenizer_time,
    ));
    progress(BarkProgressEvent::started(
        BarkProgressStage::LoadSemanticWeights,
    ));
    let start = Instant::now();
    let semantic_weights =
        load_bark_causal_transformer_weights(paths, "semantic", &config.semantic_config)?;
    let semantic_model = BarkCausalTransformer::new_with_threads(
        config.semantic_config.clone(),
        semantic_weights,
        options.threads,
    )?;
    profile.semantic_load_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::LoadSemanticWeights,
        profile.semantic_load_time,
    ));
    let semantic_input = build_semantic_input_with_history(
        &tokenizer,
        &options.text,
        generation_config,
        history_prompt.as_ref(),
    )?;
    progress(BarkProgressEvent::started(
        BarkProgressStage::GenerateSemanticTokens,
    ));
    let start = Instant::now();
    let semantic_tokens = generate_semantic_tokens_with_progress(
        &semantic_model,
        &semantic_input,
        generation_config,
        options,
        |current, total| {
            progress(BarkProgressEvent::advanced(
                BarkProgressStage::GenerateSemanticTokens,
                current,
                total,
            ));
        },
    )?;
    profile.semantic_generation_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::GenerateSemanticTokens,
        profile.semantic_generation_time,
    ));
    progress(BarkProgressEvent::started(
        BarkProgressStage::LoadCoarseWeights,
    ));
    let start = Instant::now();
    let coarse_weights = load_bark_causal_transformer_weights(
        paths,
        "coarse_acoustics",
        &config.coarse_acoustics_config,
    )?;
    let coarse_model = BarkCausalTransformer::new_with_threads(
        config.coarse_acoustics_config.clone(),
        coarse_weights,
        options.threads,
    )?;
    profile.coarse_load_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::LoadCoarseWeights,
        profile.coarse_load_time,
    ));
    let coarse_input = build_coarse_input(&semantic_tokens, generation_config)?;
    progress(BarkProgressEvent::started(
        BarkProgressStage::GenerateCoarseCodes,
    ));
    let start = Instant::now();
    let coarse_codes = generate_coarse_codes_with_history_and_progress(
        &coarse_model,
        &coarse_input,
        generation_config,
        options,
        history_prompt.as_ref(),
        |current, total| {
            progress(BarkProgressEvent::advanced(
                BarkProgressStage::GenerateCoarseCodes,
                current,
                total,
            ));
        },
    )?;
    profile.coarse_generation_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::GenerateCoarseCodes,
        profile.coarse_generation_time,
    ));
    let coarse_frames = coarse_codes.first().map(Vec::len).unwrap_or(0);
    progress(BarkProgressEvent::started(
        BarkProgressStage::LoadFineWeights,
    ));
    let start = Instant::now();
    let fine_weights = load_bark_fine_transformer_weights(paths, &config.fine_acoustics_config)?;
    let fine_model = BarkFineTransformer::new_with_threads(
        config.fine_acoustics_config.clone(),
        fine_weights,
        options.threads,
    )?;
    profile.fine_load_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::LoadFineWeights,
        profile.fine_load_time,
    ));
    if config.fine_acoustics_config.n_codes_total
        != generation_config.fine_acoustics_config.n_fine_codebooks
    {
        return Err(BarkError::InvalidConfig(format!(
            "fine_acoustics_config.n_codes_total {} does not match generation n_fine_codebooks {}",
            config.fine_acoustics_config.n_codes_total,
            generation_config.fine_acoustics_config.n_fine_codebooks
        )));
    }
    progress(BarkProgressEvent::started(
        BarkProgressStage::GenerateFineCodes,
    ));
    let start = Instant::now();
    let fine_codes = generate_fine_codes_with_history_and_progress(
        &fine_model,
        &coarse_codes,
        generation_config,
        options,
        history_prompt.as_ref(),
        |current, total| {
            progress(BarkProgressEvent::advanced(
                BarkProgressStage::GenerateFineCodes,
                current,
                total,
            ));
        },
    )?;
    profile.fine_generation_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::GenerateFineCodes,
        profile.fine_generation_time,
    ));
    progress(BarkProgressEvent::started(
        BarkProgressStage::LoadEncodecWeights,
    ));
    let start = Instant::now();
    let encodec_weights = load_bark_encodec_decoder_weights(paths, &config.codec_config)?;
    profile.encodec_load_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::LoadEncodecWeights,
        profile.encodec_load_time,
    ));
    progress(BarkProgressEvent::started(
        BarkProgressStage::DecodeEncodecAudio,
    ));
    let start = Instant::now();
    let samples = decode_encodec_audio(
        &fine_codes.codebooks,
        &config.codec_config,
        &encodec_weights,
    )?;
    profile.encodec_decode_time = start.elapsed();
    progress(BarkProgressEvent::finished(
        BarkProgressStage::DecodeEncodecAudio,
        profile.encodec_decode_time,
    ));
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(BarkError::InvalidInput(
            "native Bark EnCodec decode produced empty or non-finite audio".to_string(),
        ));
    }
    let _ = (
        info.tensor_count,
        info.f32_tensor_count,
        coarse_frames,
        fine_codes.frames(),
    );
    profile.total_time = total_start.elapsed();
    Ok(BarkGenerationTrace {
        audio: BarkAudioOutput {
            sample_rate: generation_config.sample_rate,
            samples,
        },
        semantic_tokens,
        coarse_codes,
        fine_codes: fine_codes.codebooks,
        profile,
    })
}

fn validate_bark_rust_options(options: &BarkRuntimeOptions) -> Result<()> {
    options.validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bark::{
        BarkCoarseGenerationConfig, BarkFineGenerationConfig, BarkSemanticGenerationConfig,
    };

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

    #[test]
    fn runtime_options_take_defaults_from_generation_config() {
        let generation_config = generation_config();

        let options = BarkRuntimeOptions::from_generation_config(
            "hello",
            Some("en_speaker_6".to_string()),
            7,
            &generation_config,
        );

        assert_eq!(options.text, "hello");
        assert_eq!(options.voice_preset.as_deref(), Some("en_speaker_6"));
        assert_eq!(options.seed, 7);
        assert_eq!(options.semantic_sampling.temperature, 0.7);
        assert_eq!(options.max_semantic_tokens, Some(768));
        options.validate().unwrap();
    }

    #[test]
    fn runtime_options_reject_empty_text() {
        let generation_config = generation_config();
        let options =
            BarkRuntimeOptions::from_generation_config("   ", None, 1, &generation_config);

        let err = options.validate().unwrap_err();

        assert!(matches!(err, BarkError::InvalidInput(_)));
    }

    #[test]
    fn rust_runtime_accepts_voice_preset_after_history_prompt_support() {
        let generation_config = generation_config();
        let options = BarkRuntimeOptions::from_generation_config(
            "hello",
            Some("en_speaker_6".to_string()),
            1,
            &generation_config,
        );

        validate_bark_rust_options(&options).unwrap();
    }

    #[test]
    fn runtime_profile_reports_model_stage_time() {
        let profile = BarkRuntimeProfile {
            semantic_generation_time: Duration::from_millis(1),
            coarse_generation_time: Duration::from_millis(2),
            fine_generation_time: Duration::from_millis(3),
            encodec_decode_time: Duration::from_millis(4),
            ..BarkRuntimeProfile::default()
        };

        assert_eq!(profile.model_stage_time(), Duration::from_millis(10));
    }
}
