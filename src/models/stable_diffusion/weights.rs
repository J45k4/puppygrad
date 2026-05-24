use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::models::safetensors::{read_safetensors_file, tensor_data_as_f32, TensorStore};

use super::{
    AutoencoderKlConfig, ClipTextConfig, Result, SdTensor, StableDiffusionError,
    Unet2DConditionConfig, Unet2DConditionModelWeights, UnetAttentionWeights, UnetConv2dWeights,
    UnetDownBlockWeights, UnetFeedForwardWeights, UnetLinearWeights, UnetMidBlockWeights,
    UnetResnetBlockWeights, UnetSpatialTransformerWeights, UnetTransformerBlockWeights,
    UnetUpBlockWeights, VaeAttentionBlockWeights, VaeConv2dWeights, VaeDecoderModelWeights,
    VaeMidBlockWeights, VaeResnetBlockWeights, VaeUpDecoderBlockWeights,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StableDiffusionTensorManifestRow {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub bytes: usize,
}

pub fn load_safetensors_manifest(path: &Path) -> Result<Vec<StableDiffusionTensorManifestRow>> {
    let bytes = read_safetensors_file(path).map_err(|err| {
        StableDiffusionError::Asset(format!("failed to read {}: {err}", path.display()))
    })?;
    let store = TensorStore::from_bytes(path, &bytes).map_err(|err| {
        StableDiffusionError::Asset(format!("failed to parse {}: {err}", path.display()))
    })?;
    let mut rows = Vec::new();
    for name in store.names() {
        let tensor = store.required(name).map_err(|err| {
            StableDiffusionError::Asset(format!("failed to read tensor {name}: {err}"))
        })?;
        let shape = tensor.shape().to_vec();
        rows.push(StableDiffusionTensorManifestRow {
            name: name.to_string(),
            dtype: format!("{:?}", tensor.dtype()),
            elements: shape.iter().product(),
            bytes: tensor.data().len(),
            shape,
        });
    }
    rows.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(rows)
}

pub fn write_manifest_tsv(path: &Path, rows: &[StableDiffusionTensorManifestRow]) -> Result<()> {
    let mut out = String::from("name\tdtype\tshape\telements\tbytes\n");
    for row in rows {
        out.push_str(&format!(
            "{}\t{}\t{:?}\t{}\t{}\n",
            row.name, row.dtype, row.shape, row.elements, row.bytes
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipTextWeightsManifest {
    pub tensor_count: usize,
    pub layers: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClipTextWeights {
    pub manifest: ClipTextWeightsManifest,
    pub token_embedding: Vec<f32>,
    pub position_embedding: Vec<f32>,
    pub layers: Vec<ClipTextLayerWeights>,
    pub final_layer_norm_weight: Vec<f32>,
    pub final_layer_norm_bias: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClipTextLayerWeights {
    pub self_attn: ClipTextAttentionWeights,
    pub layer_norm1_weight: Vec<f32>,
    pub layer_norm1_bias: Vec<f32>,
    pub mlp_fc1_weight: Vec<f32>,
    pub mlp_fc1_bias: Vec<f32>,
    pub mlp_fc2_weight: Vec<f32>,
    pub mlp_fc2_bias: Vec<f32>,
    pub layer_norm2_weight: Vec<f32>,
    pub layer_norm2_bias: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClipTextAttentionWeights {
    pub q_proj_weight: Vec<f32>,
    pub q_proj_bias: Vec<f32>,
    pub k_proj_weight: Vec<f32>,
    pub k_proj_bias: Vec<f32>,
    pub v_proj_weight: Vec<f32>,
    pub v_proj_bias: Vec<f32>,
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Vec<f32>,
}

pub fn load_clip_text_weights(
    path: impl AsRef<Path>,
    config: &ClipTextConfig,
) -> Result<ClipTextWeights> {
    let path = path.as_ref();
    let bytes = read_safetensors_file(path).map_err(|err| {
        StableDiffusionError::Asset(format!("failed to read {}: {err}", path.display()))
    })?;
    let store = TensorStore::from_bytes(path, &bytes).map_err(|err| {
        StableDiffusionError::Asset(format!("failed to parse {}: {err}", path.display()))
    })?;
    load_clip_text_weights_from_store(&store, config)
}

pub fn load_clip_text_weights_from_store(
    store: &TensorStore<'_>,
    config: &ClipTextConfig,
) -> Result<ClipTextWeights> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("text_model.encoder.layers.{layer}");
        layers.push(ClipTextLayerWeights {
            self_attn: load_clip_attention(
                store,
                &format!("{prefix}.self_attn"),
                config.hidden_size,
            )?,
            layer_norm1_weight: required_f32(
                store,
                &format!("{prefix}.layer_norm1.weight"),
                &[config.hidden_size],
            )?,
            layer_norm1_bias: required_f32(
                store,
                &format!("{prefix}.layer_norm1.bias"),
                &[config.hidden_size],
            )?,
            mlp_fc1_weight: required_transposed_dense_weight(
                store,
                &format!("{prefix}.mlp.fc1.weight"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            mlp_fc1_bias: required_f32(
                store,
                &format!("{prefix}.mlp.fc1.bias"),
                &[config.intermediate_size],
            )?,
            mlp_fc2_weight: required_transposed_dense_weight(
                store,
                &format!("{prefix}.mlp.fc2.weight"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            mlp_fc2_bias: required_f32(
                store,
                &format!("{prefix}.mlp.fc2.bias"),
                &[config.hidden_size],
            )?,
            layer_norm2_weight: required_f32(
                store,
                &format!("{prefix}.layer_norm2.weight"),
                &[config.hidden_size],
            )?,
            layer_norm2_bias: required_f32(
                store,
                &format!("{prefix}.layer_norm2.bias"),
                &[config.hidden_size],
            )?,
        });
    }

    Ok(ClipTextWeights {
        manifest: ClipTextWeightsManifest {
            tensor_count: 4 + config.num_hidden_layers * 16,
            layers: config.num_hidden_layers,
        },
        token_embedding: required_f32(
            store,
            "text_model.embeddings.token_embedding.weight",
            &[config.vocab_size, config.hidden_size],
        )?,
        position_embedding: required_f32(
            store,
            "text_model.embeddings.position_embedding.weight",
            &[config.max_position_embeddings, config.hidden_size],
        )?,
        layers,
        final_layer_norm_weight: required_f32(
            store,
            "text_model.final_layer_norm.weight",
            &[config.hidden_size],
        )?,
        final_layer_norm_bias: required_f32(
            store,
            "text_model.final_layer_norm.bias",
            &[config.hidden_size],
        )?,
    })
}

fn load_clip_attention(
    store: &TensorStore<'_>,
    prefix: &str,
    hidden_size: usize,
) -> Result<ClipTextAttentionWeights> {
    Ok(ClipTextAttentionWeights {
        q_proj_weight: required_transposed_dense_weight(
            store,
            &format!("{prefix}.q_proj.weight"),
            hidden_size,
            hidden_size,
        )?,
        q_proj_bias: required_f32(store, &format!("{prefix}.q_proj.bias"), &[hidden_size])?,
        k_proj_weight: required_transposed_dense_weight(
            store,
            &format!("{prefix}.k_proj.weight"),
            hidden_size,
            hidden_size,
        )?,
        k_proj_bias: required_f32(store, &format!("{prefix}.k_proj.bias"), &[hidden_size])?,
        v_proj_weight: required_transposed_dense_weight(
            store,
            &format!("{prefix}.v_proj.weight"),
            hidden_size,
            hidden_size,
        )?,
        v_proj_bias: required_f32(store, &format!("{prefix}.v_proj.bias"), &[hidden_size])?,
        out_proj_weight: required_transposed_dense_weight(
            store,
            &format!("{prefix}.out_proj.weight"),
            hidden_size,
            hidden_size,
        )?,
        out_proj_bias: required_f32(store, &format!("{prefix}.out_proj.bias"), &[hidden_size])?,
    })
}

fn required_f32(store: &TensorStore<'_>, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    store
        .required_f32_lossy(name, expected_shape)
        .map_err(|err| StableDiffusionError::Asset(err.to_string()))
}

fn required_transposed_dense_weight(
    store: &TensorStore<'_>,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let src = required_f32(store, name, &[out_features, in_features])?;
    let mut transposed = vec![0.0; src.len()];
    for out_feature in 0..out_features {
        for in_feature in 0..in_features {
            transposed[in_feature * out_features + out_feature] =
                src[out_feature * in_features + in_feature];
        }
    }
    Ok(transposed)
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedStableDiffusionTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Unet2DConditionWeights {
    pub tensors: BTreeMap<String, LoadedStableDiffusionTensor>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VaeDecoderWeights {
    pub tensors: BTreeMap<String, LoadedStableDiffusionTensor>,
}

pub fn load_unet_2d_condition_weights(path: impl AsRef<Path>) -> Result<Unet2DConditionWeights> {
    Ok(Unet2DConditionWeights {
        tensors: load_component_tensors(path.as_ref(), "UNet")?,
    })
}

pub fn load_unet_2d_condition_model_weights(
    path: impl AsRef<Path>,
    config: &Unet2DConditionConfig,
) -> Result<Unet2DConditionModelWeights> {
    let weights = load_unet_2d_condition_weights(path)?;
    unet_2d_condition_model_weights_from_tensors(&weights.tensors, config)
}

pub fn unet_2d_condition_model_weights_from_tensors(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    config: &Unet2DConditionConfig,
) -> Result<Unet2DConditionModelWeights> {
    let norm_groups = config.norm_num_groups.unwrap_or(32);
    if norm_groups == 0 {
        return Err(StableDiffusionError::Config(
            "native UNet norm_num_groups must be > 0".to_string(),
        ));
    }
    let layers_per_block = scalar_usize_value(&config.layers_per_block, "UNet layers_per_block")?;
    let block_count = config.block_out_channels.len();
    if layers_per_block == 0 {
        return Err(StableDiffusionError::Config(
            "UNet layers_per_block must be > 0".to_string(),
        ));
    }
    if config.down_block_types.len() != block_count || config.up_block_types.len() != block_count {
        return Err(StableDiffusionError::Config(format!(
            "UNet has {} channel blocks but {} down block types and {} up block types",
            block_count,
            config.down_block_types.len(),
            config.up_block_types.len()
        )));
    }
    let attention_head_dims = per_block_usize_value(
        &config.attention_head_dim,
        block_count,
        "UNet attention_head_dim",
        8,
    )?;
    let transformer_layers = per_block_usize_value(
        &config.transformer_layers_per_block,
        block_count,
        "UNet transformer_layers_per_block",
        1,
    )?;
    let first_channels = config.block_out_channels[0];
    let time_embedding_dim = first_channels * 4;

    let mut down_blocks = Vec::with_capacity(block_count);
    let mut input_channels = first_channels;
    for block_index in 0..block_count {
        let out_channels = config.block_out_channels[block_index];
        let has_cross_attention = config.down_block_types[block_index].contains("CrossAttn");
        let mut resnets = Vec::with_capacity(layers_per_block);
        let mut attentions = Vec::with_capacity(layers_per_block);
        for layer in 0..layers_per_block {
            let resnet_in_channels = if block_index == 0 && layer == 0 {
                first_channels
            } else if layer == 0 {
                input_channels
            } else {
                out_channels
            };
            resnets.push(unet_resnet_block_weights_from_tensors(
                tensors,
                &format!("down_blocks.{block_index}.resnets.{layer}"),
                resnet_in_channels,
                out_channels,
                time_embedding_dim,
            )?);
            attentions.push(if has_cross_attention {
                Some(load_unet_spatial_transformer(
                    tensors,
                    &format!("down_blocks.{block_index}.attentions.{layer}"),
                    out_channels,
                    config.cross_attention_dim,
                    attention_head_dims[block_index],
                    transformer_layers[block_index],
                )?)
            } else {
                None
            });
        }
        let downsample = if block_index + 1 == block_count {
            None
        } else {
            Some(load_unet_conv2d(
                tensors,
                &format!("down_blocks.{block_index}.downsamplers.0.conv"),
                out_channels,
                out_channels,
                3,
                2,
                1,
            )?)
        };
        down_blocks.push(UnetDownBlockWeights {
            resnets,
            attentions,
            downsample,
        });
        input_channels = out_channels;
    }

    let last_channels = *config.block_out_channels.last().ok_or_else(|| {
        StableDiffusionError::Config("UNet block_out_channels must not be empty".to_string())
    })?;
    let mid_block = UnetMidBlockWeights {
        resnet1: unet_resnet_block_weights_from_tensors(
            tensors,
            "mid_block.resnets.0",
            last_channels,
            last_channels,
            time_embedding_dim,
        )?,
        attentions: vec![load_unet_spatial_transformer(
            tensors,
            "mid_block.attentions.0",
            last_channels,
            config.cross_attention_dim,
            *attention_head_dims.last().unwrap_or(&8),
            *transformer_layers.last().unwrap_or(&1),
        )?],
        resnets: vec![unet_resnet_block_weights_from_tensors(
            tensors,
            "mid_block.resnets.1",
            last_channels,
            last_channels,
            time_embedding_dim,
        )?],
    };

    let reversed_channels = config
        .block_out_channels
        .iter()
        .copied()
        .rev()
        .collect::<Vec<_>>();
    let mut up_blocks = Vec::with_capacity(block_count);
    let mut prev_output_channels = reversed_channels[0];
    for block_index in 0..block_count {
        let out_channels = reversed_channels[block_index];
        let input_skip_channels = reversed_channels[(block_index + 1).min(block_count - 1)];
        let source_attention_index = block_count - 1 - block_index;
        let has_cross_attention = config.up_block_types[block_index].contains("CrossAttn");
        let mut resnets = Vec::with_capacity(layers_per_block + 1);
        let mut attentions = Vec::with_capacity(layers_per_block + 1);
        for layer in 0..=layers_per_block {
            let resnet_in_channels = if layer == 0 {
                prev_output_channels
            } else {
                out_channels
            };
            let skip_channels = if layer == layers_per_block {
                input_skip_channels
            } else {
                out_channels
            };
            resnets.push(unet_resnet_block_weights_from_tensors(
                tensors,
                &format!("up_blocks.{block_index}.resnets.{layer}"),
                resnet_in_channels + skip_channels,
                out_channels,
                time_embedding_dim,
            )?);
            attentions.push(if has_cross_attention {
                Some(load_unet_spatial_transformer(
                    tensors,
                    &format!("up_blocks.{block_index}.attentions.{layer}"),
                    out_channels,
                    config.cross_attention_dim,
                    attention_head_dims[source_attention_index],
                    transformer_layers[source_attention_index],
                )?)
            } else {
                None
            });
        }
        let upsample = if block_index + 1 == block_count {
            None
        } else {
            Some(load_unet_conv2d(
                tensors,
                &format!("up_blocks.{block_index}.upsamplers.0.conv"),
                out_channels,
                out_channels,
                3,
                1,
                1,
            )?)
        };
        up_blocks.push(UnetUpBlockWeights {
            resnets,
            attentions,
            upsample,
        });
        prev_output_channels = out_channels;
    }

    Ok(Unet2DConditionModelWeights {
        conv_in: load_unet_conv2d(
            tensors,
            "conv_in",
            first_channels,
            config.in_channels,
            3,
            1,
            1,
        )?,
        time_embedding_linear1: load_unet_linear(
            tensors,
            "time_embedding.linear_1",
            time_embedding_dim,
            first_channels,
        )?,
        time_embedding_linear2: load_unet_linear(
            tensors,
            "time_embedding.linear_2",
            time_embedding_dim,
            time_embedding_dim,
        )?,
        down_blocks,
        mid_block,
        up_blocks,
        conv_norm_out_weight: required_loaded_vec(
            tensors,
            "conv_norm_out.weight",
            &[first_channels],
        )?,
        conv_norm_out_bias: required_loaded_vec(tensors, "conv_norm_out.bias", &[first_channels])?,
        conv_out: load_unet_conv2d(
            tensors,
            "conv_out",
            config.out_channels,
            first_channels,
            3,
            1,
            1,
        )?,
        time_embedding_dim,
        norm_groups,
        norm_eps: 1e-5,
    })
}

pub fn unet_resnet_block_weights_from_tensors(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    time_embedding_dim: usize,
) -> Result<UnetResnetBlockWeights> {
    Ok(UnetResnetBlockWeights {
        norm1_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm1.weight"),
            &[in_channels],
        )?,
        norm1_bias: required_loaded_vec(tensors, &format!("{prefix}.norm1.bias"), &[in_channels])?,
        conv1: load_unet_conv2d(
            tensors,
            &format!("{prefix}.conv1"),
            out_channels,
            in_channels,
            3,
            1,
            1,
        )?,
        time_emb_proj: load_unet_linear(
            tensors,
            &format!("{prefix}.time_emb_proj"),
            out_channels,
            time_embedding_dim,
        )?,
        norm2_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm2.weight"),
            &[out_channels],
        )?,
        norm2_bias: required_loaded_vec(tensors, &format!("{prefix}.norm2.bias"), &[out_channels])?,
        conv2: load_unet_conv2d(
            tensors,
            &format!("{prefix}.conv2"),
            out_channels,
            out_channels,
            3,
            1,
            1,
        )?,
        shortcut: if in_channels == out_channels {
            None
        } else if tensors.contains_key(&format!("{prefix}.conv_shortcut.weight")) {
            Some(load_unet_conv2d(
                tensors,
                &format!("{prefix}.conv_shortcut"),
                out_channels,
                in_channels,
                1,
                1,
                0,
            )?)
        } else {
            return Err(StableDiffusionError::Asset(format!(
                "UNet resnet block {prefix} changes channels {in_channels}->{out_channels} but has no conv_shortcut"
            )));
        },
        output_scale_factor: 1.0,
    })
}

pub fn unet_attention_weights_from_tensors(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    query_dim: usize,
    context_dim: usize,
    inner_dim: usize,
    heads: usize,
) -> Result<UnetAttentionWeights> {
    Ok(UnetAttentionWeights {
        to_q: load_unet_linear(tensors, &format!("{prefix}.to_q"), inner_dim, query_dim)?,
        to_k: load_unet_linear(tensors, &format!("{prefix}.to_k"), inner_dim, context_dim)?,
        to_v: load_unet_linear(tensors, &format!("{prefix}.to_v"), inner_dim, context_dim)?,
        to_out: load_unet_linear(tensors, &format!("{prefix}.to_out.0"), query_dim, inner_dim)?,
        heads,
    })
}

pub fn unet_transformer_block_weights_from_tensors(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    inner_dim: usize,
    cross_attention_dim: usize,
    attention_heads: usize,
    feed_forward_dim: usize,
) -> Result<UnetTransformerBlockWeights> {
    Ok(UnetTransformerBlockWeights {
        norm1_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm1.weight"),
            &[inner_dim],
        )?,
        norm1_bias: required_loaded_vec(tensors, &format!("{prefix}.norm1.bias"), &[inner_dim])?,
        self_attn: unet_attention_weights_from_tensors(
            tensors,
            &format!("{prefix}.attn1"),
            inner_dim,
            inner_dim,
            inner_dim,
            attention_heads,
        )?,
        norm2_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm2.weight"),
            &[inner_dim],
        )?,
        norm2_bias: required_loaded_vec(tensors, &format!("{prefix}.norm2.bias"), &[inner_dim])?,
        cross_attn: unet_attention_weights_from_tensors(
            tensors,
            &format!("{prefix}.attn2"),
            inner_dim,
            cross_attention_dim,
            inner_dim,
            attention_heads,
        )?,
        norm3_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm3.weight"),
            &[inner_dim],
        )?,
        norm3_bias: required_loaded_vec(tensors, &format!("{prefix}.norm3.bias"), &[inner_dim])?,
        feed_forward: UnetFeedForwardWeights {
            geglu_proj: load_unet_linear(
                tensors,
                &format!("{prefix}.ff.net.0.proj"),
                feed_forward_dim * 2,
                inner_dim,
            )?,
            out_proj: load_unet_linear(
                tensors,
                &format!("{prefix}.ff.net.2"),
                inner_dim,
                feed_forward_dim,
            )?,
        },
    })
}

fn load_unet_spatial_transformer(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    channels: usize,
    cross_attention_dim: usize,
    attention_heads: usize,
    transformer_layers: usize,
) -> Result<UnetSpatialTransformerWeights> {
    if attention_heads == 0 || channels % attention_heads != 0 {
        return Err(StableDiffusionError::Config(format!(
            "UNet attention_head_dim {attention_heads} must divide channels {channels}"
        )));
    }
    if transformer_layers == 0 {
        return Err(StableDiffusionError::Config(
            "UNet transformer_layers_per_block must be > 0".to_string(),
        ));
    }
    let mut blocks = Vec::with_capacity(transformer_layers);
    for layer in 0..transformer_layers {
        blocks.push(unet_transformer_block_weights_from_tensors(
            tensors,
            &format!("{prefix}.transformer_blocks.{layer}"),
            channels,
            cross_attention_dim,
            attention_heads,
            channels * 4,
        )?);
    }
    Ok(UnetSpatialTransformerWeights {
        norm_weight: required_loaded_vec(tensors, &format!("{prefix}.norm.weight"), &[channels])?,
        norm_bias: required_loaded_vec(tensors, &format!("{prefix}.norm.bias"), &[channels])?,
        proj_in: load_unet_conv2d(
            tensors,
            &format!("{prefix}.proj_in"),
            channels,
            channels,
            1,
            1,
            0,
        )?,
        transformer_blocks: blocks,
        proj_out: load_unet_conv2d(
            tensors,
            &format!("{prefix}.proj_out"),
            channels,
            channels,
            1,
            1,
            0,
        )?,
    })
}

fn scalar_usize_value(value: &serde_json::Value, name: &str) -> Result<usize> {
    value
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| StableDiffusionError::Config(format!("{name} must be an unsigned integer")))
}

fn per_block_usize_value(
    value: &serde_json::Value,
    block_count: usize,
    name: &str,
    default: usize,
) -> Result<Vec<usize>> {
    if value.is_null() {
        return Ok(vec![default; block_count]);
    }
    if let Some(value) = value.as_u64() {
        return Ok(vec![value as usize; block_count]);
    }
    if let Some(values) = value.as_array() {
        if values.len() != block_count {
            return Err(StableDiffusionError::Config(format!(
                "{name} has {} entries but UNet has {block_count} blocks",
                values.len()
            )));
        }
        return values
            .iter()
            .map(|value| {
                value.as_u64().map(|value| value as usize).ok_or_else(|| {
                    StableDiffusionError::Config(format!(
                        "{name} entries must be unsigned integers"
                    ))
                })
            })
            .collect();
    }
    Err(StableDiffusionError::Config(format!(
        "{name} must be an unsigned integer or array"
    )))
}

pub fn load_vae_decoder_weights(path: impl AsRef<Path>) -> Result<VaeDecoderWeights> {
    Ok(VaeDecoderWeights {
        tensors: load_component_tensors(path.as_ref(), "VAE")?,
    })
}

pub fn load_vae_decoder_model_weights(
    path: impl AsRef<Path>,
    config: &AutoencoderKlConfig,
) -> Result<VaeDecoderModelWeights> {
    let weights = load_vae_decoder_weights(path)?;
    vae_decoder_model_weights_from_tensors(&weights.tensors, config)
}

pub fn vae_decoder_model_weights_from_tensors(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    config: &AutoencoderKlConfig,
) -> Result<VaeDecoderModelWeights> {
    let norm_groups = config.norm_num_groups.ok_or_else(|| {
        StableDiffusionError::Config(
            "native VAE decoder weight loading requires norm_num_groups".to_string(),
        )
    })?;
    if norm_groups == 0 {
        return Err(StableDiffusionError::Config(
            "native VAE decoder norm_num_groups must be > 0".to_string(),
        ));
    }
    let layers_per_block = config.layers_per_block.ok_or_else(|| {
        StableDiffusionError::Config(
            "native VAE decoder weight loading requires layers_per_block".to_string(),
        )
    })?;
    let last_channels = *config.block_out_channels.last().ok_or_else(|| {
        StableDiffusionError::Config("VAE block_out_channels must not be empty".to_string())
    })?;
    let reversed_channels = config
        .block_out_channels
        .iter()
        .copied()
        .rev()
        .collect::<Vec<_>>();

    let mut up_blocks = Vec::with_capacity(reversed_channels.len());
    for block_index in 0..reversed_channels.len() {
        let out_channels = reversed_channels[block_index];
        let block_in_channels = if block_index == 0 {
            reversed_channels[0]
        } else {
            reversed_channels[block_index - 1]
        };
        let mut resnets = Vec::with_capacity(layers_per_block + 1);
        for layer in 0..=layers_per_block {
            let in_channels = if layer == 0 {
                block_in_channels
            } else {
                out_channels
            };
            resnets.push(load_vae_resnet_block(
                tensors,
                &format!("decoder.up_blocks.{block_index}.resnets.{layer}"),
                in_channels,
                out_channels,
            )?);
        }
        let upsample = if block_index + 1 == reversed_channels.len() {
            None
        } else {
            Some(load_vae_conv2d(
                tensors,
                &format!("decoder.up_blocks.{block_index}.upsamplers.0.conv"),
                out_channels,
                out_channels,
                3,
                1,
            )?)
        };
        up_blocks.push(VaeUpDecoderBlockWeights { resnets, upsample });
    }

    Ok(VaeDecoderModelWeights {
        post_quant_conv: Some(load_vae_conv2d(
            tensors,
            "post_quant_conv",
            config.latent_channels,
            config.latent_channels,
            1,
            0,
        )?),
        conv_in: load_vae_conv2d(
            tensors,
            "decoder.conv_in",
            last_channels,
            config.latent_channels,
            3,
            1,
        )?,
        mid_block: VaeMidBlockWeights {
            resnet1: load_vae_resnet_block(
                tensors,
                "decoder.mid_block.resnets.0",
                last_channels,
                last_channels,
            )?,
            attention: Some(load_vae_attention_block(
                tensors,
                "decoder.mid_block.attentions.0",
                last_channels,
            )?),
            resnet2: load_vae_resnet_block(
                tensors,
                "decoder.mid_block.resnets.1",
                last_channels,
                last_channels,
            )?,
        },
        up_blocks,
        conv_norm_out_weight: required_loaded_vec(
            tensors,
            "decoder.conv_norm_out.weight",
            &[config.block_out_channels[0]],
        )?,
        conv_norm_out_bias: required_loaded_vec(
            tensors,
            "decoder.conv_norm_out.bias",
            &[config.block_out_channels[0]],
        )?,
        conv_out: load_vae_conv2d(
            tensors,
            "decoder.conv_out",
            3,
            config.block_out_channels[0],
            3,
            1,
        )?,
    })
}

fn load_vae_resnet_block(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
) -> Result<VaeResnetBlockWeights> {
    Ok(VaeResnetBlockWeights {
        norm1_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm1.weight"),
            &[in_channels],
        )?,
        norm1_bias: required_loaded_vec(tensors, &format!("{prefix}.norm1.bias"), &[in_channels])?,
        conv1: load_vae_conv2d(
            tensors,
            &format!("{prefix}.conv1"),
            out_channels,
            in_channels,
            3,
            1,
        )?,
        norm2_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.norm2.weight"),
            &[out_channels],
        )?,
        norm2_bias: required_loaded_vec(tensors, &format!("{prefix}.norm2.bias"), &[out_channels])?,
        conv2: load_vae_conv2d(
            tensors,
            &format!("{prefix}.conv2"),
            out_channels,
            out_channels,
            3,
            1,
        )?,
        shortcut: if in_channels == out_channels {
            None
        } else {
            load_optional_vae_shortcut(tensors, prefix, out_channels, in_channels)?
        },
    })
}

fn load_unet_conv2d(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<UnetConv2dWeights> {
    Ok(UnetConv2dWeights {
        weight: SdTensor::new(
            [out_channels, in_channels, kernel, kernel],
            required_loaded_conv_weight(
                tensors,
                &format!("{prefix}.weight"),
                out_channels,
                in_channels,
                kernel,
            )?,
        )?,
        bias: optional_loaded_vec(tensors, &format!("{prefix}.bias"), &[out_channels])?,
        stride,
        padding,
    })
}

fn load_unet_linear(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) -> Result<UnetLinearWeights> {
    Ok(UnetLinearWeights {
        weight: required_loaded_transposed_dense(
            tensors,
            &format!("{prefix}.weight"),
            out_features,
            in_features,
        )?,
        bias: optional_loaded_vec(tensors, &format!("{prefix}.bias"), &[out_features])?
            .unwrap_or_else(|| vec![0.0; out_features]),
        in_features,
        out_features,
    })
}

fn load_optional_vae_shortcut(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
) -> Result<Option<VaeConv2dWeights>> {
    for name in ["conv_shortcut", "nin_shortcut"] {
        let conv_prefix = format!("{prefix}.{name}");
        if tensors.contains_key(&format!("{conv_prefix}.weight")) {
            return load_vae_conv2d(tensors, &conv_prefix, out_channels, in_channels, 1, 0)
                .map(Some);
        }
    }
    Err(StableDiffusionError::Asset(format!(
        "VAE resnet block {prefix} changes channels {in_channels}->{out_channels} but has no conv_shortcut/nin_shortcut"
    )))
}

fn load_vae_attention_block(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    channels: usize,
) -> Result<VaeAttentionBlockWeights> {
    Ok(VaeAttentionBlockWeights {
        norm_weight: required_loaded_vec(
            tensors,
            &format!("{prefix}.group_norm.weight"),
            &[channels],
        )?,
        norm_bias: required_loaded_vec(tensors, &format!("{prefix}.group_norm.bias"), &[channels])?,
        query: load_vae_conv2d(
            tensors,
            &format!("{prefix}.query"),
            channels,
            channels,
            1,
            0,
        )?,
        key: load_vae_conv2d(tensors, &format!("{prefix}.key"), channels, channels, 1, 0)?,
        value: load_vae_conv2d(
            tensors,
            &format!("{prefix}.value"),
            channels,
            channels,
            1,
            0,
        )?,
        proj_attn: load_vae_conv2d(
            tensors,
            &format!("{prefix}.proj_attn"),
            channels,
            channels,
            1,
            0,
        )?,
    })
}

fn load_vae_conv2d(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    padding: usize,
) -> Result<VaeConv2dWeights> {
    Ok(VaeConv2dWeights {
        weight: SdTensor::new(
            [out_channels, in_channels, kernel, kernel],
            required_loaded_conv_weight(
                tensors,
                &format!("{prefix}.weight"),
                out_channels,
                in_channels,
                kernel,
            )?,
        )?,
        bias: optional_loaded_vec(tensors, &format!("{prefix}.bias"), &[out_channels])?,
        padding,
    })
}

fn required_loaded_vec(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let tensor = tensors.get(name).ok_or_else(|| {
        StableDiffusionError::Asset(format!("missing required Stable Diffusion tensor {name}"))
    })?;
    if tensor.shape != expected_shape {
        return Err(StableDiffusionError::Asset(format!(
            "tensor {name} shape {:?} does not match expected {:?}",
            tensor.shape, expected_shape
        )));
    }
    Ok(tensor.data.clone())
}

fn required_loaded_conv_weight(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let tensor = tensors.get(name).ok_or_else(|| {
        StableDiffusionError::Asset(format!("missing required Stable Diffusion tensor {name}"))
    })?;
    let conv_shape = [out_channels, in_channels, kernel, kernel];
    let linear_pointwise_shape = [out_channels, in_channels];
    if tensor.shape == conv_shape {
        return Ok(tensor.data.clone());
    }
    if kernel == 1 && tensor.shape == linear_pointwise_shape {
        return Ok(tensor.data.clone());
    }
    let expected = if kernel == 1 {
        format!("{conv_shape:?} or {linear_pointwise_shape:?}")
    } else {
        format!("{conv_shape:?}")
    };
    Err(StableDiffusionError::Asset(format!(
        "tensor {name} shape {:?} does not match expected {expected}",
        tensor.shape
    )))
}

fn optional_loaded_vec(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Option<Vec<f32>>> {
    match tensors.get(name) {
        Some(tensor) if tensor.shape == expected_shape => Ok(Some(tensor.data.clone())),
        Some(tensor) => Err(StableDiffusionError::Asset(format!(
            "tensor {name} shape {:?} does not match expected {:?}",
            tensor.shape, expected_shape
        ))),
        None => Ok(None),
    }
}

fn required_loaded_transposed_dense(
    tensors: &BTreeMap<String, LoadedStableDiffusionTensor>,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let src = required_loaded_vec(tensors, name, &[out_features, in_features])?;
    let mut transposed = vec![0.0; src.len()];
    for out_feature in 0..out_features {
        for in_feature in 0..in_features {
            transposed[in_feature * out_features + out_feature] =
                src[out_feature * in_features + in_feature];
        }
    }
    Ok(transposed)
}

fn load_component_tensors(
    path: &Path,
    component: &str,
) -> Result<BTreeMap<String, LoadedStableDiffusionTensor>> {
    let bytes = read_safetensors_file(path).map_err(|err| {
        StableDiffusionError::Asset(format!(
            "failed to read {component} weights {}: {err}",
            path.display()
        ))
    })?;
    let store = TensorStore::from_bytes(path, &bytes).map_err(|err| {
        StableDiffusionError::Asset(format!(
            "failed to parse {component} weights {}: {err}",
            path.display()
        ))
    })?;
    let mut tensors = BTreeMap::new();
    for name in store.names() {
        let tensor = store
            .required(name)
            .map_err(|err| StableDiffusionError::Asset(err.to_string()))?;
        let data = tensor_data_as_f32(name, &tensor)
            .map_err(|err| StableDiffusionError::Asset(err.to_string()))?;
        tensors.insert(
            name.to_string(),
            LoadedStableDiffusionTensor {
                shape: tensor.shape().to_vec(),
                data,
            },
        );
    }
    if tensors.is_empty() {
        return Err(StableDiffusionError::Asset(format!(
            "{component} safetensors file {} contains no tensors",
            path.display()
        )));
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::path::Path;

    fn config() -> ClipTextConfig {
        ClipTextConfig {
            class_name: "CLIPTextModel".to_string(),
            vocab_size: 3,
            hidden_size: 2,
            intermediate_size: 3,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            max_position_embeddings: 4,
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-5,
        }
    }

    fn tensor_data(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn temp_safetensors_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "puppygrad-sd-{name}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn insert_loaded(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        name: impl Into<String>,
        shape: Vec<usize>,
    ) {
        let len = shape.iter().product();
        tensors.insert(
            name.into(),
            LoadedStableDiffusionTensor {
                shape,
                data: vec![0.0; len],
            },
        );
    }

    fn insert_conv(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        out_channels: usize,
        in_channels: usize,
        kernel: usize,
    ) {
        insert_loaded(
            tensors,
            format!("{prefix}.weight"),
            vec![out_channels, in_channels, kernel, kernel],
        );
        insert_loaded(tensors, format!("{prefix}.bias"), vec![out_channels]);
    }

    fn insert_linear(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        out_features: usize,
        in_features: usize,
    ) {
        insert_loaded(
            tensors,
            format!("{prefix}.weight"),
            vec![out_features, in_features],
        );
        insert_loaded(tensors, format!("{prefix}.bias"), vec![out_features]);
    }

    fn insert_resnet(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
    ) {
        insert_loaded(tensors, format!("{prefix}.norm1.weight"), vec![in_channels]);
        insert_loaded(tensors, format!("{prefix}.norm1.bias"), vec![in_channels]);
        insert_conv(
            tensors,
            &format!("{prefix}.conv1"),
            out_channels,
            in_channels,
            3,
        );
        insert_loaded(
            tensors,
            format!("{prefix}.norm2.weight"),
            vec![out_channels],
        );
        insert_loaded(tensors, format!("{prefix}.norm2.bias"), vec![out_channels]);
        insert_conv(
            tensors,
            &format!("{prefix}.conv2"),
            out_channels,
            out_channels,
            3,
        );
        if in_channels != out_channels {
            insert_conv(
                tensors,
                &format!("{prefix}.conv_shortcut"),
                out_channels,
                in_channels,
                1,
            );
        }
    }

    fn insert_unet_resnet(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        time_embedding_dim: usize,
    ) {
        insert_resnet(tensors, prefix, in_channels, out_channels);
        insert_linear(
            tensors,
            &format!("{prefix}.time_emb_proj"),
            out_channels,
            time_embedding_dim,
        );
    }

    fn insert_attention(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        channels: usize,
    ) {
        insert_loaded(
            tensors,
            format!("{prefix}.group_norm.weight"),
            vec![channels],
        );
        insert_loaded(tensors, format!("{prefix}.group_norm.bias"), vec![channels]);
        for projection in ["query", "key", "value", "proj_attn"] {
            insert_conv(
                tensors,
                &format!("{prefix}.{projection}"),
                channels,
                channels,
                1,
            );
        }
    }

    fn insert_unet_attention_projection(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        query_dim: usize,
        context_dim: usize,
        inner_dim: usize,
    ) {
        insert_linear(tensors, &format!("{prefix}.to_q"), inner_dim, query_dim);
        insert_linear(tensors, &format!("{prefix}.to_k"), inner_dim, context_dim);
        insert_linear(tensors, &format!("{prefix}.to_v"), inner_dim, context_dim);
        insert_linear(tensors, &format!("{prefix}.to_out.0"), query_dim, inner_dim);
    }

    fn insert_unet_transformer(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        inner_dim: usize,
        cross_attention_dim: usize,
    ) {
        for norm in ["norm1", "norm2", "norm3"] {
            insert_loaded(tensors, format!("{prefix}.{norm}.weight"), vec![inner_dim]);
            insert_loaded(tensors, format!("{prefix}.{norm}.bias"), vec![inner_dim]);
        }
        insert_unet_attention_projection(
            tensors,
            &format!("{prefix}.attn1"),
            inner_dim,
            inner_dim,
            inner_dim,
        );
        insert_unet_attention_projection(
            tensors,
            &format!("{prefix}.attn2"),
            inner_dim,
            cross_attention_dim,
            inner_dim,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.ff.net.0.proj"),
            inner_dim * 8,
            inner_dim,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.ff.net.2"),
            inner_dim,
            inner_dim * 4,
        );
    }

    fn insert_unet_spatial_transformer(
        tensors: &mut BTreeMap<String, LoadedStableDiffusionTensor>,
        prefix: &str,
        channels: usize,
        cross_attention_dim: usize,
    ) {
        insert_loaded(tensors, format!("{prefix}.norm.weight"), vec![channels]);
        insert_loaded(tensors, format!("{prefix}.norm.bias"), vec![channels]);
        insert_conv(tensors, &format!("{prefix}.proj_in"), channels, channels, 1);
        insert_unet_transformer(
            tensors,
            &format!("{prefix}.transformer_blocks.0"),
            channels,
            cross_attention_dim,
        );
        insert_conv(
            tensors,
            &format!("{prefix}.proj_out"),
            channels,
            channels,
            1,
        );
    }

    #[test]
    fn loads_clip_text_weights_from_diffusers_names() -> Result<()> {
        let entries = [
            (
                "text_model.embeddings.token_embedding.weight",
                vec![3, 2],
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            (
                "text_model.embeddings.position_embedding.weight",
                vec![4, 2],
                vec![0.0; 8],
            ),
            (
                "text_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            (
                "text_model.encoder.layers.0.self_attn.q_proj.bias",
                vec![2],
                vec![0.0; 2],
            ),
            (
                "text_model.encoder.layers.0.self_attn.k_proj.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "text_model.encoder.layers.0.self_attn.k_proj.bias",
                vec![2],
                vec![0.0; 2],
            ),
            (
                "text_model.encoder.layers.0.self_attn.v_proj.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "text_model.encoder.layers.0.self_attn.v_proj.bias",
                vec![2],
                vec![0.0; 2],
            ),
            (
                "text_model.encoder.layers.0.self_attn.out_proj.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "text_model.encoder.layers.0.self_attn.out_proj.bias",
                vec![2],
                vec![0.0; 2],
            ),
            (
                "text_model.encoder.layers.0.layer_norm1.weight",
                vec![2],
                vec![1.0; 2],
            ),
            (
                "text_model.encoder.layers.0.layer_norm1.bias",
                vec![2],
                vec![0.0; 2],
            ),
            (
                "text_model.encoder.layers.0.mlp.fc1.weight",
                vec![3, 2],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            (
                "text_model.encoder.layers.0.mlp.fc1.bias",
                vec![3],
                vec![0.0; 3],
            ),
            (
                "text_model.encoder.layers.0.mlp.fc2.weight",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            (
                "text_model.encoder.layers.0.mlp.fc2.bias",
                vec![2],
                vec![0.0; 2],
            ),
            (
                "text_model.encoder.layers.0.layer_norm2.weight",
                vec![2],
                vec![1.0; 2],
            ),
            (
                "text_model.encoder.layers.0.layer_norm2.bias",
                vec![2],
                vec![0.0; 2],
            ),
            ("text_model.final_layer_norm.weight", vec![2], vec![1.0; 2]),
            ("text_model.final_layer_norm.bias", vec![2], vec![0.0; 2]),
        ];
        let buffers: Vec<Vec<u8>> = entries
            .iter()
            .map(|(_, _, values)| tensor_data(values))
            .collect();
        let views: Vec<_> = entries
            .iter()
            .zip(buffers.iter())
            .map(|((name, shape, _), data)| {
                (
                    *name,
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let bytes = serialize(views, None).unwrap();
        let store = TensorStore::from_bytes(Path::new("memory.safetensors"), &bytes)
            .map_err(|err| StableDiffusionError::Asset(err.to_string()))?;

        let weights = load_clip_text_weights_from_store(&store, &config())?;

        assert_eq!(weights.manifest.layers, 1);
        assert_eq!(weights.token_embedding, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            weights.layers[0].self_attn.q_proj_weight,
            vec![1.0, 3.0, 2.0, 4.0]
        );
        assert_eq!(
            weights.layers[0].mlp_fc1_weight,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn loads_component_weights_as_f32_tensors() -> Result<()> {
        let first = tensor_data(&[1.0, 2.0, 3.0, 4.0]);
        let second = tensor_data(&[5.0, 6.0]);
        let views = vec![
            (
                "conv.weight",
                TensorView::new(Dtype::F32, vec![1, 1, 2, 2], &first).unwrap(),
            ),
            (
                "conv.bias",
                TensorView::new(Dtype::F32, vec![2], &second).unwrap(),
            ),
        ];
        let path = temp_safetensors_path("component");
        std::fs::write(&path, serialize(views, None).unwrap()).unwrap();

        let weights = load_unet_2d_condition_weights(&path)?;

        std::fs::remove_file(&path).ok();
        assert_eq!(weights.tensors.len(), 2);
        assert_eq!(weights.tensors["conv.weight"].shape, vec![1, 1, 2, 2]);
        assert_eq!(weights.tensors["conv.bias"].data, vec![5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn maps_vae_decoder_diffusers_names_into_typed_weights() -> Result<()> {
        let config = AutoencoderKlConfig {
            class_name: "AutoencoderKL".to_string(),
            latent_channels: 1,
            block_out_channels: vec![1],
            down_block_types: vec!["DownEncoderBlock2D".to_string()],
            up_block_types: vec!["UpDecoderBlock2D".to_string()],
            layers_per_block: Some(1),
            scaling_factor: 1.0,
            norm_num_groups: Some(1),
        };
        let mut tensors = BTreeMap::new();
        insert_conv(&mut tensors, "post_quant_conv", 1, 1, 1);
        insert_conv(&mut tensors, "decoder.conv_in", 1, 1, 3);
        insert_resnet(&mut tensors, "decoder.mid_block.resnets.0", 1, 1);
        insert_attention(&mut tensors, "decoder.mid_block.attentions.0", 1);
        insert_resnet(&mut tensors, "decoder.mid_block.resnets.1", 1, 1);
        insert_resnet(&mut tensors, "decoder.up_blocks.0.resnets.0", 1, 1);
        insert_resnet(&mut tensors, "decoder.up_blocks.0.resnets.1", 1, 1);
        insert_loaded(&mut tensors, "decoder.conv_norm_out.weight", vec![1]);
        insert_loaded(&mut tensors, "decoder.conv_norm_out.bias", vec![1]);
        insert_conv(&mut tensors, "decoder.conv_out", 3, 1, 3);

        let weights = vae_decoder_model_weights_from_tensors(&tensors, &config)?;

        assert_eq!(weights.up_blocks.len(), 1);
        assert_eq!(weights.up_blocks[0].resnets.len(), 2);
        assert!(weights.mid_block.attention.is_some());
        assert_eq!(weights.conv_out.weight.shape(), &[3, 1, 3, 3]);
        Ok(())
    }

    #[test]
    fn maps_unet_resnet_and_transformer_diffusers_names() -> Result<()> {
        let mut tensors = BTreeMap::new();
        let resnet_prefix = "down_blocks.0.resnets.0";
        insert_loaded(
            &mut tensors,
            format!("{resnet_prefix}.norm1.weight"),
            vec![2],
        );
        insert_loaded(&mut tensors, format!("{resnet_prefix}.norm1.bias"), vec![2]);
        insert_conv(&mut tensors, &format!("{resnet_prefix}.conv1"), 3, 2, 3);
        insert_linear(
            &mut tensors,
            &format!("{resnet_prefix}.time_emb_proj"),
            3,
            4,
        );
        insert_loaded(
            &mut tensors,
            format!("{resnet_prefix}.norm2.weight"),
            vec![3],
        );
        insert_loaded(&mut tensors, format!("{resnet_prefix}.norm2.bias"), vec![3]);
        insert_conv(&mut tensors, &format!("{resnet_prefix}.conv2"), 3, 3, 3);
        insert_conv(
            &mut tensors,
            &format!("{resnet_prefix}.conv_shortcut"),
            3,
            2,
            1,
        );

        let block_prefix = "down_blocks.0.attentions.0.transformer_blocks.0";
        for norm in ["norm1", "norm2", "norm3"] {
            insert_loaded(
                &mut tensors,
                format!("{block_prefix}.{norm}.weight"),
                vec![2],
            );
            insert_loaded(&mut tensors, format!("{block_prefix}.{norm}.bias"), vec![2]);
        }
        for attn in ["attn1", "attn2"] {
            insert_linear(&mut tensors, &format!("{block_prefix}.{attn}.to_q"), 2, 2);
            let context_dim = if attn == "attn1" { 2 } else { 4 };
            insert_linear(
                &mut tensors,
                &format!("{block_prefix}.{attn}.to_k"),
                2,
                context_dim,
            );
            insert_linear(
                &mut tensors,
                &format!("{block_prefix}.{attn}.to_v"),
                2,
                context_dim,
            );
            insert_linear(
                &mut tensors,
                &format!("{block_prefix}.{attn}.to_out.0"),
                2,
                2,
            );
        }
        insert_linear(&mut tensors, &format!("{block_prefix}.ff.net.0.proj"), 8, 2);
        insert_linear(&mut tensors, &format!("{block_prefix}.ff.net.2"), 2, 4);

        let resnet = unet_resnet_block_weights_from_tensors(&tensors, resnet_prefix, 2, 3, 4)?;
        let transformer =
            unet_transformer_block_weights_from_tensors(&tensors, block_prefix, 2, 4, 1, 4)?;

        assert!(resnet.shortcut.is_some());
        assert_eq!(resnet.time_emb_proj.out_features, 3);
        assert_eq!(transformer.cross_attn.to_k.in_features, 4);
        assert_eq!(transformer.feed_forward.geglu_proj.out_features, 8);
        Ok(())
    }

    #[test]
    fn maps_full_unet_model_from_diffusers_names() -> Result<()> {
        let config = Unet2DConditionConfig {
            class_name: "UNet2DConditionModel".to_string(),
            sample_size: Some(64),
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![1],
            down_block_types: vec!["DownBlock2D".to_string()],
            up_block_types: vec!["UpBlock2D".to_string()],
            layers_per_block: serde_json::json!(1),
            cross_attention_dim: 2,
            attention_head_dim: serde_json::json!(1),
            transformer_layers_per_block: serde_json::json!(1),
            norm_num_groups: Some(1),
        };
        let mut tensors = BTreeMap::new();
        insert_conv(&mut tensors, "conv_in", 1, 4, 3);
        insert_linear(&mut tensors, "time_embedding.linear_1", 4, 1);
        insert_linear(&mut tensors, "time_embedding.linear_2", 4, 4);
        insert_unet_resnet(&mut tensors, "down_blocks.0.resnets.0", 1, 1, 4);
        insert_unet_resnet(&mut tensors, "mid_block.resnets.0", 1, 1, 4);
        insert_unet_spatial_transformer(&mut tensors, "mid_block.attentions.0", 1, 2);
        insert_unet_resnet(&mut tensors, "mid_block.resnets.1", 1, 1, 4);
        insert_unet_resnet(&mut tensors, "up_blocks.0.resnets.0", 2, 1, 4);
        insert_unet_resnet(&mut tensors, "up_blocks.0.resnets.1", 2, 1, 4);
        insert_loaded(&mut tensors, "conv_norm_out.weight", vec![1]);
        insert_loaded(&mut tensors, "conv_norm_out.bias", vec![1]);
        insert_conv(&mut tensors, "conv_out", 4, 1, 3);

        let weights = unet_2d_condition_model_weights_from_tensors(&tensors, &config)?;

        assert_eq!(weights.down_blocks.len(), 1);
        assert_eq!(weights.down_blocks[0].resnets.len(), 1);
        assert_eq!(weights.mid_block.attentions.len(), 1);
        assert_eq!(weights.up_blocks[0].resnets.len(), 2);
        assert_eq!(weights.conv_out.weight.shape(), &[4, 1, 3, 3]);
        Ok(())
    }
}
