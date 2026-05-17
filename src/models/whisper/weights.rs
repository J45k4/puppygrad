use std::path::Path;

use safetensors::Dtype;

use crate::models::safetensors::{
    read_safetensors_file, tensor_f32, validate_tensor, SafeTensorLoadError, TensorStore,
};

use super::{Result, WhisperConfig, WhisperError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperWeightsManifest {
    pub tensor_count: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub tied_output_projection: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperWeights {
    pub manifest: WhisperWeightsManifest,
    pub encoder: WhisperEncoderWeights,
    pub decoder: WhisperDecoderWeights,
    pub output_projection: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperEncoderWeights {
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub positional_embedding: Vec<f32>,
    pub layers: Vec<WhisperEncoderLayerWeights>,
    pub ln_w: Vec<f32>,
    pub ln_b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperEncoderLayerWeights {
    pub self_attn: WhisperAttentionWeights,
    pub self_attn_ln_w: Vec<f32>,
    pub self_attn_ln_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperDecoderWeights {
    pub token_embedding: Vec<f32>,
    pub positional_embedding: Vec<f32>,
    pub layers: Vec<WhisperDecoderLayerWeights>,
    pub ln_w: Vec<f32>,
    pub ln_b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperDecoderLayerWeights {
    pub self_attn: WhisperAttentionWeights,
    pub self_attn_ln_w: Vec<f32>,
    pub self_attn_ln_b: Vec<f32>,
    pub cross_attn: WhisperAttentionWeights,
    pub cross_attn_ln_w: Vec<f32>,
    pub cross_attn_ln_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperAttentionWeights {
    pub q_w: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Option<Vec<f32>>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

pub fn load_whisper_weights(
    path: impl AsRef<Path>,
    config: &WhisperConfig,
) -> Result<WhisperWeights> {
    let path = path.as_ref();
    let bytes = read_safetensors_file(path).map_err(whisper_tensor_error)?;
    let store = TensorStore::from_bytes(path, &bytes).map_err(whisper_tensor_error)?;
    let manifest = validate_whisper_tensor_store(&store, config)?;
    let output_projection = if manifest.tied_output_projection {
        None
    } else {
        Some(required_transposed_dense_weight(
            &store,
            "proj_out.weight",
            config.n_vocab,
            config.n_text_state,
        )?)
    };

    Ok(WhisperWeights {
        manifest,
        encoder: load_encoder_weights(&store, config)?,
        decoder: load_decoder_weights(&store, config)?,
        output_projection,
    })
}

pub fn validate_whisper_weights(
    path: impl AsRef<Path>,
    config: &WhisperConfig,
) -> Result<WhisperWeightsManifest> {
    let path = path.as_ref();
    let bytes = read_safetensors_file(path).map_err(whisper_tensor_error)?;
    let store = TensorStore::from_bytes(path, &bytes).map_err(whisper_tensor_error)?;
    validate_whisper_tensor_store(&store, config)
}

fn validate_whisper_tensor_store(
    store: &TensorStore<'_>,
    config: &WhisperConfig,
) -> Result<WhisperWeightsManifest> {
    validate_root_tensors(store, config)?;
    for layer in 0..config.n_audio_layer {
        validate_encoder_layer(store, config, layer)?;
    }
    for layer in 0..config.n_text_layer {
        validate_decoder_layer(store, config, layer)?;
    }
    let tied_output_projection = store
        .optional("proj_out.weight")
        .map_err(whisper_tensor_error)?
        .is_none();
    if !tied_output_projection {
        required_tensor(
            store,
            "proj_out.weight",
            &[config.n_vocab, config.n_text_state],
        )?;
    }
    Ok(WhisperWeightsManifest {
        tensor_count: expected_tensor_count(config, tied_output_projection),
        encoder_layers: config.n_audio_layer,
        decoder_layers: config.n_text_layer,
        tied_output_projection,
    })
}

fn load_encoder_weights(
    store: &TensorStore<'_>,
    config: &WhisperConfig,
) -> Result<WhisperEncoderWeights> {
    let mut layers = Vec::with_capacity(config.n_audio_layer);
    for layer in 0..config.n_audio_layer {
        let prefix = format!("model.encoder.layers.{layer}");
        layers.push(WhisperEncoderLayerWeights {
            self_attn: load_attention(store, &format!("{prefix}.self_attn"), config.n_audio_state)?,
            self_attn_ln_w: required_f32(
                store,
                &format!("{prefix}.self_attn_layer_norm.weight"),
                &[config.n_audio_state],
            )?,
            self_attn_ln_b: required_f32(
                store,
                &format!("{prefix}.self_attn_layer_norm.bias"),
                &[config.n_audio_state],
            )?,
            fc1_w: required_transposed_dense_weight(
                store,
                &format!("{prefix}.fc1.weight"),
                config.n_audio_mlp,
                config.n_audio_state,
            )?,
            fc1_b: required_f32(store, &format!("{prefix}.fc1.bias"), &[config.n_audio_mlp])?,
            fc2_w: required_transposed_dense_weight(
                store,
                &format!("{prefix}.fc2.weight"),
                config.n_audio_state,
                config.n_audio_mlp,
            )?,
            fc2_b: required_f32(
                store,
                &format!("{prefix}.fc2.bias"),
                &[config.n_audio_state],
            )?,
            final_ln_w: required_f32(
                store,
                &format!("{prefix}.final_layer_norm.weight"),
                &[config.n_audio_state],
            )?,
            final_ln_b: required_f32(
                store,
                &format!("{prefix}.final_layer_norm.bias"),
                &[config.n_audio_state],
            )?,
        });
    }

    Ok(WhisperEncoderWeights {
        conv1_w: required_f32(
            store,
            "model.encoder.conv1.weight",
            &[config.n_audio_state, config.n_mels, 3],
        )?,
        conv1_b: required_f32(store, "model.encoder.conv1.bias", &[config.n_audio_state])?,
        conv2_w: required_f32(
            store,
            "model.encoder.conv2.weight",
            &[config.n_audio_state, config.n_audio_state, 3],
        )?,
        conv2_b: required_f32(store, "model.encoder.conv2.bias", &[config.n_audio_state])?,
        positional_embedding: required_f32(
            store,
            "model.encoder.embed_positions.weight",
            &[config.n_audio_ctx, config.n_audio_state],
        )?,
        layers,
        ln_w: required_f32(
            store,
            "model.encoder.layer_norm.weight",
            &[config.n_audio_state],
        )?,
        ln_b: required_f32(
            store,
            "model.encoder.layer_norm.bias",
            &[config.n_audio_state],
        )?,
    })
}

fn load_decoder_weights(
    store: &TensorStore<'_>,
    config: &WhisperConfig,
) -> Result<WhisperDecoderWeights> {
    let mut layers = Vec::with_capacity(config.n_text_layer);
    for layer in 0..config.n_text_layer {
        let prefix = format!("model.decoder.layers.{layer}");
        layers.push(WhisperDecoderLayerWeights {
            self_attn: load_attention(store, &format!("{prefix}.self_attn"), config.n_text_state)?,
            self_attn_ln_w: required_f32(
                store,
                &format!("{prefix}.self_attn_layer_norm.weight"),
                &[config.n_text_state],
            )?,
            self_attn_ln_b: required_f32(
                store,
                &format!("{prefix}.self_attn_layer_norm.bias"),
                &[config.n_text_state],
            )?,
            cross_attn: load_attention(
                store,
                &format!("{prefix}.encoder_attn"),
                config.n_text_state,
            )?,
            cross_attn_ln_w: required_f32(
                store,
                &format!("{prefix}.encoder_attn_layer_norm.weight"),
                &[config.n_text_state],
            )?,
            cross_attn_ln_b: required_f32(
                store,
                &format!("{prefix}.encoder_attn_layer_norm.bias"),
                &[config.n_text_state],
            )?,
            fc1_w: required_transposed_dense_weight(
                store,
                &format!("{prefix}.fc1.weight"),
                config.n_text_mlp,
                config.n_text_state,
            )?,
            fc1_b: required_f32(store, &format!("{prefix}.fc1.bias"), &[config.n_text_mlp])?,
            fc2_w: required_transposed_dense_weight(
                store,
                &format!("{prefix}.fc2.weight"),
                config.n_text_state,
                config.n_text_mlp,
            )?,
            fc2_b: required_f32(store, &format!("{prefix}.fc2.bias"), &[config.n_text_state])?,
            final_ln_w: required_f32(
                store,
                &format!("{prefix}.final_layer_norm.weight"),
                &[config.n_text_state],
            )?,
            final_ln_b: required_f32(
                store,
                &format!("{prefix}.final_layer_norm.bias"),
                &[config.n_text_state],
            )?,
        });
    }

    Ok(WhisperDecoderWeights {
        token_embedding: required_transposed_dense_weight(
            store,
            "model.decoder.embed_tokens.weight",
            config.n_vocab,
            config.n_text_state,
        )?,
        positional_embedding: required_f32(
            store,
            "model.decoder.embed_positions.weight",
            &[config.n_text_ctx, config.n_text_state],
        )?,
        layers,
        ln_w: required_f32(
            store,
            "model.decoder.layer_norm.weight",
            &[config.n_text_state],
        )?,
        ln_b: required_f32(
            store,
            "model.decoder.layer_norm.bias",
            &[config.n_text_state],
        )?,
    })
}

fn load_attention(
    store: &TensorStore<'_>,
    prefix: &str,
    state: usize,
) -> Result<WhisperAttentionWeights> {
    Ok(WhisperAttentionWeights {
        q_w: required_transposed_dense_weight(
            store,
            &format!("{prefix}.q_proj.weight"),
            state,
            state,
        )?,
        q_b: required_f32(store, &format!("{prefix}.q_proj.bias"), &[state])?,
        k_w: required_transposed_dense_weight(
            store,
            &format!("{prefix}.k_proj.weight"),
            state,
            state,
        )?,
        k_b: optional_f32(store, &format!("{prefix}.k_proj.bias"), &[state])?,
        v_w: required_transposed_dense_weight(
            store,
            &format!("{prefix}.v_proj.weight"),
            state,
            state,
        )?,
        v_b: required_f32(store, &format!("{prefix}.v_proj.bias"), &[state])?,
        out_w: required_transposed_dense_weight(
            store,
            &format!("{prefix}.out_proj.weight"),
            state,
            state,
        )?,
        out_b: required_f32(store, &format!("{prefix}.out_proj.bias"), &[state])?,
    })
}

fn validate_root_tensors(store: &TensorStore<'_>, config: &WhisperConfig) -> Result<()> {
    required_tensor(
        store,
        "model.encoder.conv1.weight",
        &[config.n_audio_state, config.n_mels, 3],
    )?;
    required_tensor(store, "model.encoder.conv1.bias", &[config.n_audio_state])?;
    required_tensor(
        store,
        "model.encoder.conv2.weight",
        &[config.n_audio_state, config.n_audio_state, 3],
    )?;
    required_tensor(store, "model.encoder.conv2.bias", &[config.n_audio_state])?;
    required_tensor(
        store,
        "model.encoder.embed_positions.weight",
        &[config.n_audio_ctx, config.n_audio_state],
    )?;
    required_tensor(
        store,
        "model.encoder.layer_norm.weight",
        &[config.n_audio_state],
    )?;
    required_tensor(
        store,
        "model.encoder.layer_norm.bias",
        &[config.n_audio_state],
    )?;
    required_tensor(
        store,
        "model.decoder.embed_tokens.weight",
        &[config.n_vocab, config.n_text_state],
    )?;
    required_tensor(
        store,
        "model.decoder.embed_positions.weight",
        &[config.n_text_ctx, config.n_text_state],
    )?;
    required_tensor(
        store,
        "model.decoder.layer_norm.weight",
        &[config.n_text_state],
    )?;
    required_tensor(
        store,
        "model.decoder.layer_norm.bias",
        &[config.n_text_state],
    )?;
    Ok(())
}

fn validate_encoder_layer(
    store: &TensorStore<'_>,
    config: &WhisperConfig,
    layer: usize,
) -> Result<()> {
    let prefix = format!("model.encoder.layers.{layer}");
    for projection in ["q_proj", "v_proj", "out_proj"] {
        required_tensor(
            store,
            &format!("{prefix}.self_attn.{projection}.weight"),
            &[config.n_audio_state, config.n_audio_state],
        )?;
        required_tensor(
            store,
            &format!("{prefix}.self_attn.{projection}.bias"),
            &[config.n_audio_state],
        )?;
    }
    required_tensor(
        store,
        &format!("{prefix}.self_attn.k_proj.weight"),
        &[config.n_audio_state, config.n_audio_state],
    )?;
    required_tensor(
        store,
        &format!("{prefix}.self_attn_layer_norm.weight"),
        &[config.n_audio_state],
    )?;
    required_tensor(
        store,
        &format!("{prefix}.self_attn_layer_norm.bias"),
        &[config.n_audio_state],
    )?;
    required_tensor(
        store,
        &format!("{prefix}.fc1.weight"),
        &[config.n_audio_mlp, config.n_audio_state],
    )?;
    required_tensor(store, &format!("{prefix}.fc1.bias"), &[config.n_audio_mlp])?;
    required_tensor(
        store,
        &format!("{prefix}.fc2.weight"),
        &[config.n_audio_state, config.n_audio_mlp],
    )?;
    required_tensor(
        store,
        &format!("{prefix}.fc2.bias"),
        &[config.n_audio_state],
    )?;
    required_tensor(
        store,
        &format!("{prefix}.final_layer_norm.weight"),
        &[config.n_audio_state],
    )?;
    required_tensor(
        store,
        &format!("{prefix}.final_layer_norm.bias"),
        &[config.n_audio_state],
    )?;
    Ok(())
}

fn validate_decoder_layer(
    store: &TensorStore<'_>,
    config: &WhisperConfig,
    layer: usize,
) -> Result<()> {
    let prefix = format!("model.decoder.layers.{layer}");
    validate_decoder_attention(store, config, &format!("{prefix}.self_attn"))?;
    validate_decoder_attention(store, config, &format!("{prefix}.encoder_attn"))?;
    for norm in [
        "self_attn_layer_norm",
        "encoder_attn_layer_norm",
        "final_layer_norm",
    ] {
        required_tensor(
            store,
            &format!("{prefix}.{norm}.weight"),
            &[config.n_text_state],
        )?;
        required_tensor(
            store,
            &format!("{prefix}.{norm}.bias"),
            &[config.n_text_state],
        )?;
    }
    required_tensor(
        store,
        &format!("{prefix}.fc1.weight"),
        &[config.n_text_mlp, config.n_text_state],
    )?;
    required_tensor(store, &format!("{prefix}.fc1.bias"), &[config.n_text_mlp])?;
    required_tensor(
        store,
        &format!("{prefix}.fc2.weight"),
        &[config.n_text_state, config.n_text_mlp],
    )?;
    required_tensor(store, &format!("{prefix}.fc2.bias"), &[config.n_text_state])?;
    Ok(())
}

fn validate_decoder_attention(
    store: &TensorStore<'_>,
    config: &WhisperConfig,
    prefix: &str,
) -> Result<()> {
    for projection in ["q_proj", "v_proj", "out_proj"] {
        required_tensor(
            store,
            &format!("{prefix}.{projection}.weight"),
            &[config.n_text_state, config.n_text_state],
        )?;
        required_tensor(
            store,
            &format!("{prefix}.{projection}.bias"),
            &[config.n_text_state],
        )?;
    }
    required_tensor(
        store,
        &format!("{prefix}.k_proj.weight"),
        &[config.n_text_state, config.n_text_state],
    )?;
    Ok(())
}

fn required_tensor(store: &TensorStore<'_>, name: &str, expected_shape: &[usize]) -> Result<()> {
    let tensor = store.required(name).map_err(whisper_tensor_error)?;
    validate_tensor(name, &tensor, Dtype::F32, expected_shape).map_err(whisper_tensor_error)
}

fn required_f32(store: &TensorStore<'_>, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    tensor_f32(store, name, expected_shape).map_err(whisper_tensor_error)
}

fn required_transposed_dense_weight(
    store: &TensorStore<'_>,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    // Hugging Face/PyTorch linear weights are stored as [out, in], which is
    // the transposed layout used directly by Puppygrad's row-wise dot kernels.
    required_f32(store, name, &[out_features, in_features])
}

fn optional_f32(
    store: &TensorStore<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Option<Vec<f32>>> {
    store
        .optional_f32(name, expected_shape)
        .map_err(whisper_tensor_error)
}

fn expected_tensor_count(config: &WhisperConfig, tied_output_projection: bool) -> usize {
    let root = 11;
    let encoder = config.n_audio_layer * 15;
    let decoder = config.n_text_layer * 24;
    root + encoder + decoder + usize::from(!tied_output_projection)
}

fn whisper_tensor_error(err: SafeTensorLoadError) -> WhisperError {
    match err {
        SafeTensorLoadError::WrongShape {
            name,
            actual,
            expected,
        } => WhisperError::InvalidWeights(format!(
            "tensor {name} shape {actual:?} does not match expected {expected:?}"
        )),
        err => WhisperError::Asset(err.to_string()),
    }
}
