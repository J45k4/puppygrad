mod assets;
mod coarse;
mod config;
mod encodec;
mod error;
mod fine;
mod history;
mod kernels;
#[cfg(test)]
mod parity;
mod runtime;
mod semantic;
mod tokenizer;
mod transformer;
mod weights;

pub use assets::{
    default_bark_dir, prepare_bark_assets, prepare_bark_metadata_assets,
    validate_bark_native_assets, BarkAssetPaths, BARK_ASSETS, BARK_CONFIG_JSON,
    BARK_GENERATION_CONFIG_JSON, BARK_METADATA_ASSETS, BARK_NATIVE_WEIGHTS, BARK_PYTORCH_WEIGHTS,
    BARK_SMALL_MODEL_ID, BARK_SPEAKER_EMBEDDINGS_JSON, BARK_SPECIAL_TOKENS_MAP_JSON,
    BARK_TOKENIZER_CONFIG_JSON, BARK_VOCAB_TXT,
};
pub use coarse::{
    build_coarse_input, build_coarse_window_input, coarse_output_token_count,
    deinterleave_coarse_codes, generate_coarse_codes, generate_coarse_codes_with_history,
    generate_coarse_codes_with_history_and_progress, mask_coarse_logits_for_codebook,
    BarkCoarseInput,
};
pub use config::{
    load_bark_config, load_bark_generation_config, BarkCoarseGenerationConfig, BarkCodecConfig,
    BarkConfig, BarkFineGenerationConfig, BarkFineSubModelConfig, BarkGenerationConfig,
    BarkSemanticGenerationConfig, BarkSubModelConfig,
};
pub use encodec::{
    acoustic_codes_to_quantized_latents, decode_encodec_audio, decode_quantized_latents,
    BarkEncodecConvTransposeWeights, BarkEncodecConvWeights, BarkEncodecDecoderWeights,
    BarkEncodecLstmLayerWeights, BarkEncodecLstmWeights, BarkEncodecResnetBlockWeights,
    BarkEncodecUpsampleBlockWeights, BarkQuantizedLatents,
};
pub use error::{BarkError, Result};
pub use fine::{
    build_fine_code_matrix, flatten_code_matrix_with_offsets, generate_fine_codes,
    generate_fine_codes_with_history, generate_fine_codes_with_history_and_progress,
    mask_fine_logits, BarkFineCodeMatrix, BarkFineTransformer,
};
pub use history::{load_bark_history_prompt, BarkHistoryPrompt};
pub use kernels::{
    add_in_place, causal_self_attention, causal_self_attention_with_mask, embedding_lookup,
    full_self_attention, full_self_attention_with_mask, gelu_in_place, layer_norm_in_place, linear,
};
pub use runtime::{
    generate_bark_rust, generate_bark_rust_trace, generate_bark_rust_trace_with_progress,
    inspect_bark_rust_runtime, BarkAudioOutput, BarkGenerationTrace, BarkNativeRuntimeInfo,
    BarkProgressEvent, BarkProgressStage, BarkProgressStatus, BarkRuntimeOptions,
    BarkRuntimeProfile, BarkSamplingOptions,
};
pub use semantic::{
    build_semantic_input, build_semantic_input_embeddings, build_semantic_input_with_history,
    generate_semantic_tokens, generate_semantic_tokens_with_progress, mask_semantic_logits,
    BarkSemanticInput,
};
pub use tokenizer::{BarkTextEncoding, BarkTokenizer};
pub use transformer::BarkCausalTransformer;
pub use weights::{
    expected_bark_native_tensor_names, expected_causal_transformer_tensor_names,
    expected_encodec_decoder_tensor_names, expected_fine_transformer_tensor_names,
    load_bark_causal_transformer_weights, load_bark_encodec_decoder_weights,
    load_bark_encodec_decoder_weights_from_store, load_bark_fine_transformer_weights,
    load_bark_native_weight_manifest, load_bark_native_weight_manifest_file,
    BarkCausalTransformerLayerWeights, BarkCausalTransformerWeights, BarkCoarseTransformerWeights,
    BarkFineTransformerWeights, BarkNativeWeightManifest, BarkSemanticTransformerWeights,
};
