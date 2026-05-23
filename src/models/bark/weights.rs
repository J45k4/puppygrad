use std::path::Path;

use safetensors::Dtype;

use crate::models::safetensors::{parse_safetensors, read_safetensors_file, TensorStore};

use super::{
    validate_bark_native_assets, BarkAssetPaths, BarkCodecConfig, BarkConfig,
    BarkEncodecConvTransposeWeights, BarkEncodecConvWeights, BarkEncodecDecoderWeights,
    BarkEncodecLstmLayerWeights, BarkEncodecLstmWeights, BarkEncodecResnetBlockWeights,
    BarkEncodecUpsampleBlockWeights, BarkError, BarkFineSubModelConfig, BarkSubModelConfig, Result,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkNativeWeightManifest {
    pub tensor_count: usize,
    pub f32_tensor_count: usize,
    pub tensor_names: Vec<String>,
}

pub fn load_bark_native_weight_manifest(
    paths: &BarkAssetPaths,
) -> Result<BarkNativeWeightManifest> {
    validate_bark_native_assets(paths)?;
    load_bark_native_weight_manifest_file(&paths.native_weights)
}

pub fn load_bark_native_weight_manifest_file(
    path: impl AsRef<Path>,
) -> Result<BarkNativeWeightManifest> {
    let path = path.as_ref();
    let bytes = read_safetensors_file(path).map_err(|err| BarkError::Asset(err.to_string()))?;
    let tensors =
        parse_safetensors(path, &bytes).map_err(|err| BarkError::Asset(err.to_string()))?;
    let mut tensor_names = tensors
        .names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    tensor_names.sort();
    let f32_tensor_count = tensors
        .iter()
        .filter(|(_, tensor)| tensor.dtype() == Dtype::F32)
        .count();

    Ok(BarkNativeWeightManifest {
        tensor_count: tensors.len(),
        f32_tensor_count,
        tensor_names,
    })
}

pub fn load_bark_causal_transformer_weights(
    paths: &BarkAssetPaths,
    prefix: &str,
    config: &BarkSubModelConfig,
) -> Result<BarkCausalTransformerWeights> {
    validate_bark_native_assets(paths)?;
    let bytes = read_safetensors_file(&paths.native_weights)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    let store = TensorStore::from_bytes(&paths.native_weights, &bytes)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    BarkCausalTransformerWeights::load_from_store(&store, prefix, config)
}

pub fn load_bark_fine_transformer_weights(
    paths: &BarkAssetPaths,
    config: &BarkFineSubModelConfig,
) -> Result<BarkFineTransformerWeights> {
    validate_bark_native_assets(paths)?;
    let bytes = read_safetensors_file(&paths.native_weights)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    let store = TensorStore::from_bytes(&paths.native_weights, &bytes)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    BarkFineTransformerWeights::load_from_store(&store, "fine_acoustics", config)
}

pub fn load_bark_encodec_decoder_weights(
    paths: &BarkAssetPaths,
    config: &BarkCodecConfig,
) -> Result<BarkEncodecDecoderWeights> {
    validate_bark_native_assets(paths)?;
    let bytes = read_safetensors_file(&paths.native_weights)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    let store = TensorStore::from_bytes(&paths.native_weights, &bytes)
        .map_err(|err| BarkError::Asset(err.to_string()))?;
    load_bark_encodec_decoder_weights_from_store(&store, config)
}

pub fn load_bark_encodec_decoder_weights_from_store(
    store: &TensorStore<'_>,
    config: &BarkCodecConfig,
) -> Result<BarkEncodecDecoderWeights> {
    let mut quantizer_codebooks = Vec::with_capacity(config.num_quantizers);
    for codebook in 0..config.num_quantizers {
        quantizer_codebooks.push(required_f32(
            store,
            &format!("codec_model.quantizer.layers.{codebook}.codebook.embed"),
            &[config.codebook_size, config.codebook_dim],
        )?);
    }

    let scaling = 1usize << config.upsampling_ratios.len();
    let mut layer_idx = 0usize;
    let initial_channels = scaling * config.num_filters;
    let initial = load_encodec_conv1d(
        store,
        &format!("codec_model.decoder.layers.{layer_idx}.conv"),
        config.hidden_size,
        initial_channels,
        config.kernel_size,
        1,
        1,
    )?;
    layer_idx += 1;
    let lstm = load_encodec_lstm(
        store,
        &format!("codec_model.decoder.layers.{layer_idx}.lstm"),
        initial_channels,
        config.num_lstm_layers,
    )?;
    layer_idx += 1;

    let mut current_scaling = scaling;
    let mut upsample_blocks = Vec::with_capacity(config.upsampling_ratios.len());
    for &ratio in &config.upsampling_ratios {
        layer_idx += 1; // ELU
        let current_channels = current_scaling * config.num_filters;
        let next_channels = current_channels / 2;
        let upsample = load_encodec_conv_transpose1d(
            store,
            &format!("codec_model.decoder.layers.{layer_idx}.conv"),
            current_channels,
            next_channels,
            ratio * 2,
            ratio,
        )?;
        layer_idx += 1;

        let mut residual_blocks = Vec::with_capacity(config.num_residual_layers);
        for residual in 0..config.num_residual_layers {
            let dilation = config.dilation_growth_rate.pow(residual as u32);
            residual_blocks.push(load_encodec_resnet_block(
                store,
                &format!("codec_model.decoder.layers.{layer_idx}"),
                next_channels,
                config.residual_kernel_size,
                dilation,
                config.compress,
                config.use_conv_shortcut,
            )?);
            layer_idx += 1;
        }
        upsample_blocks.push(BarkEncodecUpsampleBlockWeights {
            upsample,
            residual_blocks,
        });
        current_scaling /= 2;
    }

    layer_idx += 1; // ELU
    let final_conv = load_encodec_conv1d(
        store,
        &format!("codec_model.decoder.layers.{layer_idx}.conv"),
        config.num_filters,
        config.audio_channels,
        config.last_kernel_size,
        1,
        1,
    )?;

    Ok(BarkEncodecDecoderWeights {
        quantizer_codebooks,
        initial,
        lstm,
        upsample_blocks,
        final_conv,
    })
}

pub fn expected_bark_native_tensor_names(config: &BarkConfig) -> Vec<String> {
    let mut names = Vec::new();
    names.extend(expected_causal_transformer_tensor_names(
        "semantic",
        &config.semantic_config,
    ));
    names.extend(expected_causal_transformer_tensor_names(
        "coarse_acoustics",
        &config.coarse_acoustics_config,
    ));
    names.extend(expected_fine_transformer_tensor_names(
        "fine_acoustics",
        &config.fine_acoustics_config,
    ));
    names.extend(expected_encodec_decoder_tensor_names(&config.codec_config));
    names.sort();
    names
}

pub fn expected_fine_transformer_tensor_names(
    prefix: &str,
    config: &BarkFineSubModelConfig,
) -> Vec<String> {
    let base = &config.base;
    let mut names = vec![
        format!("{prefix}.position_embeds_layer.weight"),
        format!("{prefix}.layernorm_final.weight"),
        format!("{prefix}.layernorm_final.bias"),
    ];
    for codebook in 0..config.n_codes_total {
        names.push(format!("{prefix}.input_embeds_layers.{codebook}.weight"));
    }
    for head in 0..config.n_codes_total.saturating_sub(config.n_codes_given) {
        if base.input_vocab_size != base.output_vocab_size {
            names.push(format!("{prefix}.lm_heads.{head}.weight"));
        }
    }
    for layer in 0..base.num_layers {
        let layer_prefix = format!("{prefix}.layers.{layer}");
        names.extend([
            format!("{layer_prefix}.layernorm_1.weight"),
            format!("{layer_prefix}.layernorm_1.bias"),
            format!("{layer_prefix}.attn.att_proj.weight"),
            format!("{layer_prefix}.attn.out_proj.weight"),
            format!("{layer_prefix}.layernorm_2.weight"),
            format!("{layer_prefix}.layernorm_2.bias"),
            format!("{layer_prefix}.mlp.in_proj.weight"),
            format!("{layer_prefix}.mlp.out_proj.weight"),
        ]);
        if base.bias {
            names.extend([
                format!("{layer_prefix}.attn.att_proj.bias"),
                format!("{layer_prefix}.attn.out_proj.bias"),
                format!("{layer_prefix}.mlp.in_proj.bias"),
                format!("{layer_prefix}.mlp.out_proj.bias"),
            ]);
        }
    }
    names.sort();
    names
}

pub fn expected_causal_transformer_tensor_names(
    prefix: &str,
    config: &BarkSubModelConfig,
) -> Vec<String> {
    let mut names = vec![
        format!("{prefix}.input_embeds_layer.weight"),
        format!("{prefix}.position_embeds_layer.weight"),
        format!("{prefix}.layernorm_final.weight"),
    ];
    if config.input_vocab_size != config.output_vocab_size {
        names.push(format!("{prefix}.lm_head.weight"));
    }
    if config.bias {
        names.push(format!("{prefix}.layernorm_final.bias"));
    }
    for layer in 0..config.num_layers {
        let layer_prefix = format!("{prefix}.layers.{layer}");
        names.extend([
            format!("{layer_prefix}.layernorm_1.weight"),
            format!("{layer_prefix}.attn.att_proj.weight"),
            format!("{layer_prefix}.attn.out_proj.weight"),
            format!("{layer_prefix}.layernorm_2.weight"),
            format!("{layer_prefix}.mlp.in_proj.weight"),
            format!("{layer_prefix}.mlp.out_proj.weight"),
        ]);
        if config.bias {
            names.extend([
                format!("{layer_prefix}.layernorm_1.bias"),
                format!("{layer_prefix}.attn.att_proj.bias"),
                format!("{layer_prefix}.attn.out_proj.bias"),
                format!("{layer_prefix}.layernorm_2.bias"),
                format!("{layer_prefix}.mlp.in_proj.bias"),
                format!("{layer_prefix}.mlp.out_proj.bias"),
            ]);
        }
    }
    names.sort();
    names
}

pub fn expected_encodec_decoder_tensor_names(config: &BarkCodecConfig) -> Vec<String> {
    let mut names = Vec::new();
    for codebook in 0..config.num_quantizers {
        names.push(format!(
            "codec_model.quantizer.layers.{codebook}.codebook.embed"
        ));
    }

    let scaling = 1usize << config.upsampling_ratios.len();
    let mut layer_idx = 0usize;
    names.extend(weight_norm_names(&format!(
        "codec_model.decoder.layers.{layer_idx}.conv"
    )));
    layer_idx += 1;
    names.extend(lstm_names(
        &format!("codec_model.decoder.layers.{layer_idx}.lstm"),
        config.num_lstm_layers,
    ));
    layer_idx += 1;

    let mut current_scaling = scaling;
    for &ratio in &config.upsampling_ratios {
        let _ = ratio;
        layer_idx += 1; // ELU
        names.extend(weight_norm_names(&format!(
            "codec_model.decoder.layers.{layer_idx}.conv"
        )));
        layer_idx += 1;
        for _ in 0..config.num_residual_layers {
            let block_prefix = format!("codec_model.decoder.layers.{layer_idx}");
            names.extend(weight_norm_names(&format!("{block_prefix}.block.1.conv")));
            names.extend(weight_norm_names(&format!("{block_prefix}.block.3.conv")));
            if config.use_conv_shortcut {
                names.extend(weight_norm_names(&format!("{block_prefix}.shortcut.conv")));
            }
            layer_idx += 1;
        }
        current_scaling /= 2;
        let _ = current_scaling;
    }

    layer_idx += 1; // ELU
    names.extend(weight_norm_names(&format!(
        "codec_model.decoder.layers.{layer_idx}.conv"
    )));
    names.sort();
    names
}

fn weight_norm_names(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}.weight_g"),
        format!("{prefix}.weight_v"),
        format!("{prefix}.bias"),
    ]
}

fn lstm_names(prefix: &str, layers: usize) -> Vec<String> {
    let mut names = Vec::new();
    for layer in 0..layers {
        names.extend([
            format!("{prefix}.weight_ih_l{layer}"),
            format!("{prefix}.weight_hh_l{layer}"),
            format!("{prefix}.bias_ih_l{layer}"),
            format!("{prefix}.bias_hh_l{layer}"),
        ]);
    }
    names
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkSemanticTransformerWeights {
    pub causal: BarkCausalTransformerWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkCoarseTransformerWeights {
    pub causal: BarkCausalTransformerWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkFineTransformerWeights {
    pub input_embeddings: Vec<Vec<f32>>,
    pub positional_embedding: Vec<f32>,
    pub layers: Vec<BarkCausalTransformerLayerWeights>,
    pub final_layer_norm_weight: Vec<f32>,
    pub final_layer_norm_bias: Vec<f32>,
    pub lm_head_weights: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkCausalTransformerWeights {
    pub token_embedding: Vec<f32>,
    pub positional_embedding: Vec<f32>,
    pub layers: Vec<BarkCausalTransformerLayerWeights>,
    pub final_layer_norm_weight: Vec<f32>,
    pub final_layer_norm_bias: Option<Vec<f32>>,
    pub lm_head_weight: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkCausalTransformerLayerWeights {
    pub attention_layer_norm_weight: Vec<f32>,
    pub attention_layer_norm_bias: Option<Vec<f32>>,
    pub attention_qkv_weight: Vec<f32>,
    pub attention_qkv_bias: Option<Vec<f32>>,
    pub attention_output_weight: Vec<f32>,
    pub attention_output_bias: Option<Vec<f32>>,
    pub mlp_layer_norm_weight: Vec<f32>,
    pub mlp_layer_norm_bias: Option<Vec<f32>>,
    pub mlp_in_weight: Vec<f32>,
    pub mlp_in_bias: Option<Vec<f32>>,
    pub mlp_out_weight: Vec<f32>,
    pub mlp_out_bias: Option<Vec<f32>>,
}

impl BarkCausalTransformerWeights {
    pub fn load_from_store(
        store: &TensorStore<'_>,
        prefix: &str,
        config: &BarkSubModelConfig,
    ) -> Result<Self> {
        validate_no_unused_transformer_layers(store, prefix, config.num_layers)?;
        let hidden = config.hidden_size;
        let mlp_hidden = hidden * 4;
        let token_embedding = required_f32(
            store,
            &format!("{prefix}.input_embeds_layer.weight"),
            &[config.input_vocab_size, hidden],
        )?;
        let positional_embedding = required_f32(
            store,
            &format!("{prefix}.position_embeds_layer.weight"),
            &[config.block_size, hidden],
        )?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let prefix = format!("{prefix}.layers.{layer}");
            layers.push(BarkCausalTransformerLayerWeights {
                attention_layer_norm_weight: required_f32(
                    store,
                    &format!("{prefix}.layernorm_1.weight"),
                    &[hidden],
                )?,
                attention_layer_norm_bias: optional_bias(
                    store,
                    &format!("{prefix}.layernorm_1.bias"),
                    config.bias,
                    &[hidden],
                )?,
                attention_qkv_weight: required_f32(
                    store,
                    &format!("{prefix}.attn.att_proj.weight"),
                    &[hidden * 3, hidden],
                )?,
                attention_qkv_bias: optional_bias(
                    store,
                    &format!("{prefix}.attn.att_proj.bias"),
                    config.bias,
                    &[hidden * 3],
                )?,
                attention_output_weight: required_f32(
                    store,
                    &format!("{prefix}.attn.out_proj.weight"),
                    &[hidden, hidden],
                )?,
                attention_output_bias: optional_bias(
                    store,
                    &format!("{prefix}.attn.out_proj.bias"),
                    config.bias,
                    &[hidden],
                )?,
                mlp_layer_norm_weight: required_f32(
                    store,
                    &format!("{prefix}.layernorm_2.weight"),
                    &[hidden],
                )?,
                mlp_layer_norm_bias: optional_bias(
                    store,
                    &format!("{prefix}.layernorm_2.bias"),
                    config.bias,
                    &[hidden],
                )?,
                mlp_in_weight: required_f32(
                    store,
                    &format!("{prefix}.mlp.in_proj.weight"),
                    &[mlp_hidden, hidden],
                )?,
                mlp_in_bias: optional_bias(
                    store,
                    &format!("{prefix}.mlp.in_proj.bias"),
                    config.bias,
                    &[mlp_hidden],
                )?,
                mlp_out_weight: required_f32(
                    store,
                    &format!("{prefix}.mlp.out_proj.weight"),
                    &[hidden, mlp_hidden],
                )?,
                mlp_out_bias: optional_bias(
                    store,
                    &format!("{prefix}.mlp.out_proj.bias"),
                    config.bias,
                    &[hidden],
                )?,
            });
        }
        let lm_head_weight = load_lm_head_weight(store, prefix, config, &token_embedding)?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            layers,
            final_layer_norm_weight: required_f32(
                store,
                &format!("{prefix}.layernorm_final.weight"),
                &[hidden],
            )?,
            final_layer_norm_bias: optional_bias(
                store,
                &format!("{prefix}.layernorm_final.bias"),
                config.bias,
                &[hidden],
            )?,
            lm_head_weight,
        })
    }
}

impl BarkFineTransformerWeights {
    pub fn load_from_store(
        store: &TensorStore<'_>,
        prefix: &str,
        config: &BarkFineSubModelConfig,
    ) -> Result<Self> {
        validate_no_unused_transformer_layers(store, prefix, config.base.num_layers)?;
        let base = &config.base;
        let hidden = base.hidden_size;
        let mlp_hidden = hidden * 4;
        let mut input_embeddings = Vec::with_capacity(config.n_codes_total);
        for codebook in 0..config.n_codes_total {
            input_embeddings.push(required_f32(
                store,
                &format!("{prefix}.input_embeds_layers.{codebook}.weight"),
                &[base.input_vocab_size, hidden],
            )?);
        }
        let positional_embedding = required_f32(
            store,
            &format!("{prefix}.position_embeds_layer.weight"),
            &[base.block_size, hidden],
        )?;
        let mut layers = Vec::with_capacity(base.num_layers);
        for layer in 0..base.num_layers {
            let prefix = format!("{prefix}.layers.{layer}");
            layers.push(BarkCausalTransformerLayerWeights {
                attention_layer_norm_weight: required_f32(
                    store,
                    &format!("{prefix}.layernorm_1.weight"),
                    &[hidden],
                )?,
                attention_layer_norm_bias: Some(required_f32(
                    store,
                    &format!("{prefix}.layernorm_1.bias"),
                    &[hidden],
                )?),
                attention_qkv_weight: required_f32(
                    store,
                    &format!("{prefix}.attn.att_proj.weight"),
                    &[hidden * 3, hidden],
                )?,
                attention_qkv_bias: optional_bias(
                    store,
                    &format!("{prefix}.attn.att_proj.bias"),
                    base.bias,
                    &[hidden * 3],
                )?,
                attention_output_weight: required_f32(
                    store,
                    &format!("{prefix}.attn.out_proj.weight"),
                    &[hidden, hidden],
                )?,
                attention_output_bias: optional_bias(
                    store,
                    &format!("{prefix}.attn.out_proj.bias"),
                    base.bias,
                    &[hidden],
                )?,
                mlp_layer_norm_weight: required_f32(
                    store,
                    &format!("{prefix}.layernorm_2.weight"),
                    &[hidden],
                )?,
                mlp_layer_norm_bias: Some(required_f32(
                    store,
                    &format!("{prefix}.layernorm_2.bias"),
                    &[hidden],
                )?),
                mlp_in_weight: required_f32(
                    store,
                    &format!("{prefix}.mlp.in_proj.weight"),
                    &[mlp_hidden, hidden],
                )?,
                mlp_in_bias: optional_bias(
                    store,
                    &format!("{prefix}.mlp.in_proj.bias"),
                    base.bias,
                    &[mlp_hidden],
                )?,
                mlp_out_weight: required_f32(
                    store,
                    &format!("{prefix}.mlp.out_proj.weight"),
                    &[hidden, mlp_hidden],
                )?,
                mlp_out_bias: optional_bias(
                    store,
                    &format!("{prefix}.mlp.out_proj.bias"),
                    base.bias,
                    &[hidden],
                )?,
            });
        }
        let mut lm_head_weights =
            Vec::with_capacity(config.n_codes_total.saturating_sub(config.n_codes_given));
        for head in 0..config.n_codes_total.saturating_sub(config.n_codes_given) {
            lm_head_weights.push(load_fine_lm_head_weight(
                store,
                prefix,
                config,
                head,
                &input_embeddings,
            )?);
        }
        Ok(Self {
            input_embeddings,
            positional_embedding,
            layers,
            final_layer_norm_weight: required_f32(
                store,
                &format!("{prefix}.layernorm_final.weight"),
                &[hidden],
            )?,
            final_layer_norm_bias: required_f32(
                store,
                &format!("{prefix}.layernorm_final.bias"),
                &[hidden],
            )?,
            lm_head_weights,
        })
    }
}

fn load_fine_lm_head_weight(
    store: &TensorStore<'_>,
    prefix: &str,
    config: &BarkFineSubModelConfig,
    head: usize,
    input_embeddings: &[Vec<f32>],
) -> Result<Vec<f32>> {
    let base = &config.base;
    let name = format!("{prefix}.lm_heads.{head}.weight");
    if let Some(weight) = store
        .optional_f32(&name, &[base.output_vocab_size, base.hidden_size])
        .map_err(|err| BarkError::Asset(err.to_string()))?
    {
        return Ok(weight);
    }
    if base.input_vocab_size == base.output_vocab_size {
        let tied_embedding = head + 1;
        if let Some(weight) = input_embeddings.get(tied_embedding) {
            return Ok(weight.clone());
        }
    }
    Err(BarkError::Asset(format!(
        "missing tensor {name}; tied fine acoustic LM heads require matching input/output vocab sizes and an input embedding for head {}",
        head + 1
    )))
}

fn load_lm_head_weight(
    store: &TensorStore<'_>,
    prefix: &str,
    config: &BarkSubModelConfig,
    token_embedding: &[f32],
) -> Result<Vec<f32>> {
    let name = format!("{prefix}.lm_head.weight");
    if let Some(weight) = store
        .optional_f32(&name, &[config.output_vocab_size, config.hidden_size])
        .map_err(|err| BarkError::Asset(err.to_string()))?
    {
        return Ok(weight);
    }
    if config.input_vocab_size == config.output_vocab_size {
        return Ok(token_embedding.to_vec());
    }
    Err(BarkError::Asset(format!(
        "missing tensor {name}; tied input/output embeddings require matching input/output vocab sizes, got {} and {}",
        config.input_vocab_size, config.output_vocab_size
    )))
}

fn validate_no_unused_transformer_layers(
    store: &TensorStore<'_>,
    prefix: &str,
    expected_layers: usize,
) -> Result<()> {
    let layer_prefix = format!("{prefix}.layers.");
    let mut unexpected = Vec::new();
    for name in store.names() {
        let Some(rest) = name.strip_prefix(&layer_prefix) else {
            continue;
        };
        let Some((layer, _)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = layer.parse::<usize>() else {
            continue;
        };
        if layer >= expected_layers && !unexpected.iter().any(|seen| seen == name) {
            unexpected.push(name.to_string());
        }
    }
    unexpected.sort();
    if unexpected.is_empty() {
        Ok(())
    } else {
        Err(BarkError::Asset(format!(
            "{prefix} checkpoint contains unused transformer layer groups beyond configured num_layers={expected_layers}: {unexpected:?}"
        )))
    }
}

fn required_f32(store: &TensorStore<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    store
        .required_f32(name, shape)
        .map_err(|err| BarkError::Asset(err.to_string()))
}

fn optional_bias(
    store: &TensorStore<'_>,
    name: &str,
    enabled: bool,
    shape: &[usize],
) -> Result<Option<Vec<f32>>> {
    if enabled {
        return required_f32(store, name, shape).map(Some);
    }
    store
        .optional_f32(name, shape)
        .map_err(|err| BarkError::Asset(err.to_string()))
}

fn load_encodec_resnet_block(
    store: &TensorStore<'_>,
    prefix: &str,
    channels: usize,
    residual_kernel_size: usize,
    dilation: usize,
    compress: usize,
    use_conv_shortcut: bool,
) -> Result<BarkEncodecResnetBlockWeights> {
    let hidden = channels / compress;
    Ok(BarkEncodecResnetBlockWeights {
        first: load_encodec_conv1d(
            store,
            &format!("{prefix}.block.1.conv"),
            channels,
            hidden,
            residual_kernel_size,
            1,
            dilation,
        )?,
        second: load_encodec_conv1d(
            store,
            &format!("{prefix}.block.3.conv"),
            hidden,
            channels,
            1,
            1,
            1,
        )?,
        shortcut: if use_conv_shortcut {
            Some(load_encodec_conv1d(
                store,
                &format!("{prefix}.shortcut.conv"),
                channels,
                channels,
                1,
                1,
                1,
            )?)
        } else {
            None
        },
    })
}

fn load_encodec_conv1d(
    store: &TensorStore<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> Result<BarkEncodecConvWeights> {
    Ok(BarkEncodecConvWeights {
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        weight: load_encodec_weight_norm(store, prefix, &[out_channels, in_channels, kernel_size])?,
        bias: required_f32(store, &format!("{prefix}.bias"), &[out_channels])?,
    })
}

fn load_encodec_conv_transpose1d(
    store: &TensorStore<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<BarkEncodecConvTransposeWeights> {
    Ok(BarkEncodecConvTransposeWeights {
        in_channels,
        out_channels,
        kernel_size,
        stride,
        weight: load_encodec_weight_norm(store, prefix, &[in_channels, out_channels, kernel_size])?,
        bias: required_f32(store, &format!("{prefix}.bias"), &[out_channels])?,
    })
}

fn load_encodec_lstm(
    store: &TensorStore<'_>,
    prefix: &str,
    channels: usize,
    layers: usize,
) -> Result<BarkEncodecLstmWeights> {
    let mut loaded_layers = Vec::with_capacity(layers);
    for layer in 0..layers {
        loaded_layers.push(BarkEncodecLstmLayerWeights {
            weight_ih: required_f32(
                store,
                &format!("{prefix}.weight_ih_l{layer}"),
                &[channels * 4, channels],
            )?,
            weight_hh: required_f32(
                store,
                &format!("{prefix}.weight_hh_l{layer}"),
                &[channels * 4, channels],
            )?,
            bias_ih: required_f32(
                store,
                &format!("{prefix}.bias_ih_l{layer}"),
                &[channels * 4],
            )?,
            bias_hh: required_f32(
                store,
                &format!("{prefix}.bias_hh_l{layer}"),
                &[channels * 4],
            )?,
        });
    }
    Ok(BarkEncodecLstmWeights {
        layers: loaded_layers,
    })
}

fn load_encodec_weight_norm(
    store: &TensorStore<'_>,
    prefix: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    if let Some(weight) = store
        .optional_f32(&format!("{prefix}.weight"), shape)
        .map_err(|err| BarkError::Asset(err.to_string()))?
    {
        return Ok(weight);
    }

    let dim0 = shape[0];
    let original0 = format!("{prefix}.parametrizations.weight.original0");
    let original1 = format!("{prefix}.parametrizations.weight.original1");
    if let Some(g) = store
        .optional_f32(&original0, &[dim0, 1, 1])
        .map_err(|err| BarkError::Asset(err.to_string()))?
    {
        let v = required_f32(store, &original1, shape)?;
        return normalize_weight_norm(&g, &v, shape);
    }

    let weight_g = format!("{prefix}.weight_g");
    let weight_v = format!("{prefix}.weight_v");
    let g = required_f32(store, &weight_g, &[dim0, 1, 1])?;
    let v = required_f32(store, &weight_v, shape)?;
    normalize_weight_norm(&g, &v, shape)
}

fn normalize_weight_norm(g: &[f32], v: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let dim0 = shape[0];
    let slice_len = shape[1..].iter().product::<usize>();
    if g.len() != dim0 || v.len() != dim0 * slice_len {
        return Err(BarkError::Asset(format!(
            "weight_norm tensor lengths do not match shape {shape:?}"
        )));
    }
    let mut weight = vec![0.0; v.len()];
    for row in 0..dim0 {
        let start = row * slice_len;
        let norm = v[start..start + slice_len]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(BarkError::Asset(format!(
                "weight_norm row {row} has invalid norm {norm}"
            )));
        }
        let scale = g[row] / norm;
        for offset in 0..slice_len {
            weight[start + offset] = v[start + offset] * scale;
        }
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};

    #[test]
    fn loads_native_manifest_from_safetensors_bytes() -> Result<()> {
        let first = [1.0f32, 2.0];
        let first_data = first
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let second = [3.0f32];
        let second_data = second
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let bytes = serialize(
            [
                (
                    "semantic.input_embeds.weight",
                    TensorView::new(Dtype::F32, vec![2], &first_data).unwrap(),
                ),
                (
                    "codec.decoder.bias",
                    TensorView::new(Dtype::F32, vec![1], &second_data).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "puppygrad-bark-manifest-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();

        let manifest = load_bark_native_weight_manifest_file(&path)?;
        std::fs::remove_file(&path).unwrap();

        assert_eq!(manifest.tensor_count, 2);
        assert_eq!(manifest.f32_tensor_count, 2);
        assert_eq!(
            manifest.tensor_names,
            ["codec.decoder.bias", "semantic.input_embeds.weight"]
        );
        Ok(())
    }

    #[test]
    fn loads_tiny_causal_transformer_weights() -> Result<()> {
        let config = tiny_submodel_config(true);
        let bytes = tiny_causal_safetensors("semantic", &config);
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes)
            .map_err(|err| BarkError::Asset(err.to_string()))?;

        let weights = BarkCausalTransformerWeights::load_from_store(&store, "semantic", &config)?;

        assert_eq!(weights.token_embedding.len(), 6 * 4);
        assert_eq!(weights.positional_embedding.len(), 8 * 4);
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.layers[0].attention_qkv_weight.len(), 12 * 4);
        assert_eq!(weights.layers[0].mlp_out_weight.len(), 4 * 16);
        assert_eq!(weights.final_layer_norm_bias.as_ref().unwrap().len(), 4);
        assert_eq!(weights.lm_head_weight.len(), 7 * 4);
        Ok(())
    }

    #[test]
    fn loads_tied_lm_head_when_output_vocab_matches_input_vocab() -> Result<()> {
        let mut config = tiny_submodel_config(true);
        config.output_vocab_size = config.input_vocab_size;
        let bytes = tiny_causal_safetensors_without_lm_head("semantic", &config);
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes)
            .map_err(|err| BarkError::Asset(err.to_string()))?;

        let weights = BarkCausalTransformerWeights::load_from_store(&store, "semantic", &config)?;

        assert_eq!(weights.lm_head_weight, weights.token_embedding);
        Ok(())
    }

    #[test]
    fn missing_lm_head_requires_tied_vocab_shape() {
        let config = tiny_submodel_config(true);
        let bytes = tiny_causal_safetensors_without_lm_head("semantic", &config);
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes).unwrap();

        let err =
            BarkCausalTransformerWeights::load_from_store(&store, "semantic", &config).unwrap_err();

        assert!(err
            .to_string()
            .contains("missing tensor semantic.lm_head.weight"));
        assert!(err.to_string().contains("tied input/output embeddings"));
    }

    #[test]
    fn rejects_unused_transformer_layer_groups() {
        let config = tiny_submodel_config(true);
        let bytes = tiny_causal_safetensors_with_unused_layer("semantic", &config);
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes).unwrap();

        let err =
            BarkCausalTransformerWeights::load_from_store(&store, "semantic", &config).unwrap_err();

        assert!(err.to_string().contains("unused transformer layer groups"));
        assert!(err
            .to_string()
            .contains("semantic.layers.1.layernorm_1.weight"));
    }

    #[test]
    fn causal_transformer_weights_report_missing_tensor() {
        let config = tiny_submodel_config(true);
        let data = f32_bytes(24);
        let bytes = serialize(
            [(
                "semantic.input_embeds_layer.weight",
                TensorView::new(Dtype::F32, vec![6, 4], &data).unwrap(),
            )],
            None,
        )
        .unwrap();
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes).unwrap();

        let err =
            BarkCausalTransformerWeights::load_from_store(&store, "semantic", &config).unwrap_err();

        assert!(err.to_string().contains("position_embeds_layer.weight"));
    }

    #[test]
    fn loads_tiny_encodec_decoder_weights_with_weight_norm() -> Result<()> {
        let config = tiny_codec_config();
        let bytes = tiny_encodec_safetensors(&config);
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes)
            .map_err(|err| BarkError::Asset(err.to_string()))?;

        let weights = load_bark_encodec_decoder_weights_from_store(&store, &config)?;

        assert_eq!(weights.quantizer_codebooks.len(), 1);
        assert_eq!(weights.quantizer_codebooks[0].len(), 2 * 2);
        assert_eq!(weights.initial.out_channels, 2);
        assert_eq!(weights.lstm.layers.len(), 1);
        assert_eq!(weights.upsample_blocks.len(), 1);
        assert_eq!(weights.upsample_blocks[0].upsample.stride, 2);
        assert_eq!(weights.upsample_blocks[0].residual_blocks.len(), 1);
        assert_eq!(weights.final_conv.out_channels, 1);
        Ok(())
    }

    #[test]
    fn loads_tiny_fine_transformer_weights() -> Result<()> {
        let config = tiny_fine_submodel_config();
        let bytes = tiny_fine_safetensors("fine_acoustics", &config);
        let store = TensorStore::from_bytes(Path::new("tiny.safetensors"), &bytes)
            .map_err(|err| BarkError::Asset(err.to_string()))?;

        let weights =
            BarkFineTransformerWeights::load_from_store(&store, "fine_acoustics", &config)?;

        assert_eq!(weights.input_embeddings.len(), 4);
        assert_eq!(weights.input_embeddings[0].len(), 6 * 4);
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.final_layer_norm_bias.len(), 4);
        assert_eq!(weights.lm_head_weights.len(), 2);
        assert_eq!(weights.lm_head_weights[0].len(), 7 * 4);
        Ok(())
    }

    #[test]
    fn expected_namespace_lists_transformers_and_encodec_tensors() {
        let config = tiny_bark_config();

        let names = expected_bark_native_tensor_names(&config);

        assert!(names.contains(&"semantic.input_embeds_layer.weight".to_string()));
        assert!(names.contains(&"coarse_acoustics.layers.0.attn.att_proj.weight".to_string()));
        assert!(names.contains(&"fine_acoustics.input_embeds_layers.0.weight".to_string()));
        assert!(names.contains(&"fine_acoustics.lm_heads.0.weight".to_string()));
        assert!(names.contains(&"codec_model.quantizer.layers.0.codebook.embed".to_string()));
        assert!(names.contains(&"codec_model.decoder.layers.0.conv.weight_g".to_string()));
        assert!(names.contains(&"codec_model.decoder.layers.1.lstm.weight_ih_l0".to_string()));
    }

    #[test]
    fn loads_local_bark_small_native_weight_smoke_if_model_exists() -> Result<()> {
        use crate::models::bark::{load_bark_config, BarkAssetPaths};

        let paths = BarkAssetPaths::new("models/bark-small");
        if !paths.native_weights.exists() || !paths.config.exists() {
            return Ok(());
        }

        let config = load_bark_config(&paths.config)?;
        let manifest = load_bark_native_weight_manifest(&paths)?;
        let expected = expected_bark_native_tensor_names(&config);
        let missing = expected
            .iter()
            .filter(|name| !manifest.tensor_names.contains(name))
            .collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "local Bark safetensors checkpoint is missing expected tensors: {missing:?}"
        );

        load_bark_causal_transformer_weights(&paths, "semantic", &config.semantic_config)?;
        load_bark_causal_transformer_weights(
            &paths,
            "coarse_acoustics",
            &config.coarse_acoustics_config,
        )?;
        load_bark_fine_transformer_weights(&paths, &config.fine_acoustics_config)?;
        load_bark_encodec_decoder_weights(&paths, &config.codec_config)?;
        Ok(())
    }

    fn tiny_submodel_config(bias: bool) -> BarkSubModelConfig {
        BarkSubModelConfig {
            block_size: 8,
            input_vocab_size: 6,
            output_vocab_size: 7,
            num_layers: 1,
            num_heads: 2,
            hidden_size: 4,
            dropout: 0.0,
            bias,
            use_cache: true,
            model_type: Some("semantic".to_string()),
        }
    }

    fn tiny_codec_config() -> BarkCodecConfig {
        BarkCodecConfig {
            sampling_rate: 24_000,
            audio_channels: 1,
            hidden_size: 2,
            num_filters: 1,
            num_residual_layers: 1,
            codebook_size: 2,
            num_quantizers: 1,
            codebook_dim: 2,
            upsampling_ratios: vec![2],
            kernel_size: 1,
            last_kernel_size: 1,
            residual_kernel_size: 1,
            dilation_growth_rate: 2,
            compress: 1,
            num_lstm_layers: 1,
            use_causal_conv: true,
            trim_right_ratio: 1.0,
            norm_type: "weight_norm".to_string(),
            pad_mode: "constant".to_string(),
            use_conv_shortcut: true,
            model_type: Some("encodec".to_string()),
        }
    }

    fn tiny_bark_config() -> BarkConfig {
        BarkConfig {
            semantic_config: tiny_submodel_config(true),
            coarse_acoustics_config: tiny_submodel_config(true),
            fine_acoustics_config: tiny_fine_submodel_config(),
            codec_config: tiny_codec_config(),
            initializer_range: 0.02,
            model_type: Some("bark".to_string()),
        }
    }

    fn tiny_fine_submodel_config() -> super::super::BarkFineSubModelConfig {
        super::super::BarkFineSubModelConfig {
            base: tiny_submodel_config(true),
            n_codes_total: 4,
            n_codes_given: 2,
        }
    }

    fn tiny_causal_safetensors(prefix: &str, config: &BarkSubModelConfig) -> Vec<u8> {
        tiny_causal_safetensors_with_options(prefix, config, true, false)
    }

    fn tiny_causal_safetensors_without_lm_head(
        prefix: &str,
        config: &BarkSubModelConfig,
    ) -> Vec<u8> {
        tiny_causal_safetensors_with_options(prefix, config, false, false)
    }

    fn tiny_causal_safetensors_with_unused_layer(
        prefix: &str,
        config: &BarkSubModelConfig,
    ) -> Vec<u8> {
        tiny_causal_safetensors_with_options(prefix, config, true, true)
    }

    fn tiny_causal_safetensors_with_options(
        prefix: &str,
        config: &BarkSubModelConfig,
        include_lm_head: bool,
        include_unused_layer: bool,
    ) -> Vec<u8> {
        let hidden = config.hidden_size;
        let mlp_hidden = hidden * 4;
        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
            (
                format!("{prefix}.input_embeds_layer.weight"),
                vec![config.input_vocab_size, hidden],
                f32_bytes(config.input_vocab_size * hidden),
            ),
            (
                format!("{prefix}.position_embeds_layer.weight"),
                vec![config.block_size, hidden],
                f32_bytes(config.block_size * hidden),
            ),
            (
                format!("{prefix}.layernorm_final.weight"),
                vec![hidden],
                f32_bytes(hidden),
            ),
            (
                format!("{prefix}.layernorm_final.bias"),
                vec![hidden],
                f32_bytes(hidden),
            ),
        ];
        if include_lm_head {
            entries.push((
                format!("{prefix}.lm_head.weight"),
                vec![config.output_vocab_size, hidden],
                f32_bytes(config.output_vocab_size * hidden),
            ));
        }
        for layer in 0..config.num_layers {
            let layer_prefix = format!("{prefix}.layers.{layer}");
            entries.extend([
                (
                    format!("{layer_prefix}.layernorm_1.weight"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.layernorm_1.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.attn.att_proj.weight"),
                    vec![hidden * 3, hidden],
                    f32_bytes(hidden * 3 * hidden),
                ),
                (
                    format!("{layer_prefix}.attn.att_proj.bias"),
                    vec![hidden * 3],
                    f32_bytes(hidden * 3),
                ),
                (
                    format!("{layer_prefix}.attn.out_proj.weight"),
                    vec![hidden, hidden],
                    f32_bytes(hidden * hidden),
                ),
                (
                    format!("{layer_prefix}.attn.out_proj.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.layernorm_2.weight"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.layernorm_2.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.in_proj.weight"),
                    vec![mlp_hidden, hidden],
                    f32_bytes(mlp_hidden * hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.in_proj.bias"),
                    vec![mlp_hidden],
                    f32_bytes(mlp_hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.out_proj.weight"),
                    vec![hidden, mlp_hidden],
                    f32_bytes(hidden * mlp_hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.out_proj.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
            ]);
        }
        if include_unused_layer {
            entries.push((
                format!("{prefix}.layers.{}.layernorm_1.weight", config.num_layers),
                vec![hidden],
                f32_bytes(hidden),
            ));
        }

        let views = entries
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize(views, None).unwrap()
    }

    fn tiny_fine_safetensors(
        prefix: &str,
        config: &super::super::BarkFineSubModelConfig,
    ) -> Vec<u8> {
        let base = &config.base;
        let hidden = base.hidden_size;
        let mlp_hidden = hidden * 4;
        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for codebook in 0..config.n_codes_total {
            entries.push((
                format!("{prefix}.input_embeds_layers.{codebook}.weight"),
                vec![base.input_vocab_size, hidden],
                f32_bytes(base.input_vocab_size * hidden),
            ));
        }
        entries.extend([
            (
                format!("{prefix}.position_embeds_layer.weight"),
                vec![base.block_size, hidden],
                f32_bytes(base.block_size * hidden),
            ),
            (
                format!("{prefix}.layernorm_final.weight"),
                vec![hidden],
                f32_bytes(hidden),
            ),
            (
                format!("{prefix}.layernorm_final.bias"),
                vec![hidden],
                f32_bytes(hidden),
            ),
        ]);
        for head in 0..config.n_codes_total - config.n_codes_given {
            entries.push((
                format!("{prefix}.lm_heads.{head}.weight"),
                vec![base.output_vocab_size, hidden],
                f32_bytes(base.output_vocab_size * hidden),
            ));
        }
        for layer in 0..base.num_layers {
            let layer_prefix = format!("{prefix}.layers.{layer}");
            entries.extend([
                (
                    format!("{layer_prefix}.layernorm_1.weight"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.layernorm_1.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.attn.att_proj.weight"),
                    vec![hidden * 3, hidden],
                    f32_bytes(hidden * 3 * hidden),
                ),
                (
                    format!("{layer_prefix}.attn.att_proj.bias"),
                    vec![hidden * 3],
                    f32_bytes(hidden * 3),
                ),
                (
                    format!("{layer_prefix}.attn.out_proj.weight"),
                    vec![hidden, hidden],
                    f32_bytes(hidden * hidden),
                ),
                (
                    format!("{layer_prefix}.attn.out_proj.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.layernorm_2.weight"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.layernorm_2.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.in_proj.weight"),
                    vec![mlp_hidden, hidden],
                    f32_bytes(mlp_hidden * hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.in_proj.bias"),
                    vec![mlp_hidden],
                    f32_bytes(mlp_hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.out_proj.weight"),
                    vec![hidden, mlp_hidden],
                    f32_bytes(hidden * mlp_hidden),
                ),
                (
                    format!("{layer_prefix}.mlp.out_proj.bias"),
                    vec![hidden],
                    f32_bytes(hidden),
                ),
            ]);
        }

        let views = entries
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize(views, None).unwrap()
    }

    fn tiny_encodec_safetensors(config: &BarkCodecConfig) -> Vec<u8> {
        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = vec![(
            "codec_model.quantizer.layers.0.codebook.embed".to_string(),
            vec![config.codebook_size, config.codebook_dim],
            f32_bytes(config.codebook_size * config.codebook_dim),
        )];
        add_weight_norm_conv(
            &mut entries,
            "codec_model.decoder.layers.0.conv",
            &[2, 2, 1],
            2,
        );
        add_lstm(&mut entries, "codec_model.decoder.layers.1.lstm", 2, 1);
        add_weight_norm_conv(
            &mut entries,
            "codec_model.decoder.layers.3.conv",
            &[2, 1, 4],
            1,
        );
        add_weight_norm_conv(
            &mut entries,
            "codec_model.decoder.layers.4.block.1.conv",
            &[1, 1, 1],
            1,
        );
        add_weight_norm_conv(
            &mut entries,
            "codec_model.decoder.layers.4.block.3.conv",
            &[1, 1, 1],
            1,
        );
        add_weight_norm_conv(
            &mut entries,
            "codec_model.decoder.layers.4.shortcut.conv",
            &[1, 1, 1],
            1,
        );
        add_weight_norm_conv(
            &mut entries,
            "codec_model.decoder.layers.6.conv",
            &[1, 1, 1],
            1,
        );

        let views = entries
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize(views, None).unwrap()
    }

    fn add_weight_norm_conv(
        entries: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        shape: &[usize],
        bias_len: usize,
    ) {
        entries.push((
            format!("{prefix}.parametrizations.weight.original0"),
            vec![shape[0], 1, 1],
            f32_constant_bytes(shape[0], 1.0),
        ));
        entries.push((
            format!("{prefix}.parametrizations.weight.original1"),
            shape.to_vec(),
            f32_bytes(shape.iter().product()),
        ));
        entries.push((
            format!("{prefix}.bias"),
            vec![bias_len],
            f32_constant_bytes(bias_len, 0.0),
        ));
    }

    fn add_lstm(
        entries: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        channels: usize,
        layers: usize,
    ) {
        for layer in 0..layers {
            entries.push((
                format!("{prefix}.weight_ih_l{layer}"),
                vec![channels * 4, channels],
                f32_constant_bytes(channels * 4 * channels, 0.0),
            ));
            entries.push((
                format!("{prefix}.weight_hh_l{layer}"),
                vec![channels * 4, channels],
                f32_constant_bytes(channels * 4 * channels, 0.0),
            ));
            entries.push((
                format!("{prefix}.bias_ih_l{layer}"),
                vec![channels * 4],
                f32_constant_bytes(channels * 4, 0.0),
            ));
            entries.push((
                format!("{prefix}.bias_hh_l{layer}"),
                vec![channels * 4],
                f32_constant_bytes(channels * 4, 0.0),
            ));
        }
    }

    fn f32_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .flat_map(|idx| ((idx + 1) as f32 / 100.0).to_le_bytes())
            .collect()
    }

    fn f32_constant_bytes(len: usize, value: f32) -> Vec<u8> {
        (0..len).flat_map(|_| value.to_le_bytes()).collect()
    }
}
