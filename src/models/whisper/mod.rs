mod assets;
mod audio;
mod config;
mod error;
mod features;
mod model;
mod preprocessor;
mod runtime;
mod tokenizer;
mod weights;

pub use assets::{default_whisper_dir, prepare_whisper_assets, WhisperAssetPaths, WHISPER_ASSETS};
pub use audio::{load_wav_pcm, load_wav_pcm_bytes, PcmAudio};
pub use config::{
    load_whisper_config, WhisperConfig, WhisperSize, WHISPER_AUDIO_CTX, WHISPER_ENGLISH_VOCAB,
    WHISPER_LARGE_V3_VOCAB, WHISPER_MULTILINGUAL_VOCAB, WHISPER_TEXT_CTX,
};
pub use error::{Result, WhisperError};
pub use features::{log_mel_spectrogram, pad_or_trim, LogMelSpectrogram};
pub use model::{
    decoder_logits, decoder_logits_with_rust_config,
    decoder_logits_with_rust_config_and_quantized_logits, encode_audio,
    encode_audio_with_rust_config, generate_greedy, generate_greedy_with_rust_config,
    generate_greedy_with_rust_config_and_quantized_logits, EncodedAudio, WhisperBackendConfig,
    WhisperBackendName, WhisperDecoderKvCache, WhisperDecoderLayerKvCache, WhisperOperationProfile,
    WhisperRustConfig,
};
pub use preprocessor::{
    load_whisper_preprocessor_config, WhisperPreprocessorConfig, WHISPER_CHUNK_SECONDS,
    WHISPER_HOP_LENGTH, WHISPER_N_FFT, WHISPER_N_FRAMES, WHISPER_N_SAMPLES, WHISPER_SAMPLE_RATE,
};
pub use runtime::WhisperRuntime;
pub use tokenizer::{WhisperSpecialTokens, WhisperTask, WhisperTokenizer};
pub use weights::{
    load_whisper_weights, validate_whisper_weights, WhisperWeights, WhisperWeightsManifest,
};
