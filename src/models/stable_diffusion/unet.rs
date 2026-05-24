use super::{
    concat_tensors, conv2d_nchw, group_norm_nchw, group_norm_silu_nchw, layer_norm_last_dim,
    linear2d, linear_flattened_last_dim, scaled_dot_product_attention, upsample_nearest2d_nchw,
    Conv2dOptions, Result, SdTensor, StableDiffusionError,
};

#[derive(Clone, Debug, PartialEq)]
pub struct UnetConv2dWeights {
    pub weight: SdTensor,
    pub bias: Option<Vec<f32>>,
    pub stride: usize,
    pub padding: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetLinearWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetResnetBlockWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub conv1: UnetConv2dWeights,
    pub time_emb_proj: UnetLinearWeights,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub conv2: UnetConv2dWeights,
    pub shortcut: Option<UnetConv2dWeights>,
    pub output_scale_factor: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetAttentionWeights {
    pub to_q: UnetLinearWeights,
    pub to_k: UnetLinearWeights,
    pub to_v: UnetLinearWeights,
    pub to_out: UnetLinearWeights,
    pub heads: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetFeedForwardWeights {
    pub geglu_proj: UnetLinearWeights,
    pub out_proj: UnetLinearWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetTransformerBlockWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub self_attn: UnetAttentionWeights,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub cross_attn: UnetAttentionWeights,
    pub norm3_weight: Vec<f32>,
    pub norm3_bias: Vec<f32>,
    pub feed_forward: UnetFeedForwardWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetSpatialTransformerWeights {
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub proj_in: UnetConv2dWeights,
    pub transformer_blocks: Vec<UnetTransformerBlockWeights>,
    pub proj_out: UnetConv2dWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetDownBlockWeights {
    pub resnets: Vec<UnetResnetBlockWeights>,
    pub attentions: Vec<Option<UnetSpatialTransformerWeights>>,
    pub downsample: Option<UnetConv2dWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetMidBlockWeights {
    pub resnet1: UnetResnetBlockWeights,
    pub attentions: Vec<UnetSpatialTransformerWeights>,
    pub resnets: Vec<UnetResnetBlockWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnetUpBlockWeights {
    pub resnets: Vec<UnetResnetBlockWeights>,
    pub attentions: Vec<Option<UnetSpatialTransformerWeights>>,
    pub upsample: Option<UnetConv2dWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Unet2DConditionModelWeights {
    pub conv_in: UnetConv2dWeights,
    pub time_embedding_linear1: UnetLinearWeights,
    pub time_embedding_linear2: UnetLinearWeights,
    pub down_blocks: Vec<UnetDownBlockWeights>,
    pub mid_block: UnetMidBlockWeights,
    pub up_blocks: Vec<UnetUpBlockWeights>,
    pub conv_norm_out_weight: Vec<f32>,
    pub conv_norm_out_bias: Vec<f32>,
    pub conv_out: UnetConv2dWeights,
    pub time_embedding_dim: usize,
    pub norm_groups: usize,
    pub norm_eps: f32,
}

pub fn timestep_embedding(timestep: usize, dim: usize, max_period: f32) -> Result<SdTensor> {
    if dim == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "timestep embedding dimension must be > 0".to_string(),
        ));
    }
    if !max_period.is_finite() || max_period <= 0.0 {
        return Err(StableDiffusionError::InvalidInput(
            "timestep embedding max_period must be finite and > 0".to_string(),
        ));
    }
    let half = dim / 2;
    let mut data = Vec::with_capacity(dim);
    for index in 0..half {
        let exponent = -(max_period.ln()) * index as f32 / half.max(1) as f32;
        let value = timestep as f32 * exponent.exp();
        data.push(value.cos());
    }
    for index in 0..half {
        let exponent = -(max_period.ln()) * index as f32 / half.max(1) as f32;
        let value = timestep as f32 * exponent.exp();
        data.push(value.sin());
    }
    if dim % 2 == 1 {
        data.push(0.0);
    }
    SdTensor::new([1, dim], data)
}

pub fn classifier_free_guidance(
    unconditional: &SdTensor,
    conditional: &SdTensor,
    guidance_scale: f32,
) -> Result<SdTensor> {
    if !guidance_scale.is_finite() {
        return Err(StableDiffusionError::InvalidInput(
            "guidance scale must be finite".to_string(),
        ));
    }
    if unconditional.shape() != conditional.shape() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "classifier-free guidance shape mismatch: unconditional {:?}, conditional {:?}",
            unconditional.shape(),
            conditional.shape()
        )));
    }
    let data = unconditional
        .data()
        .iter()
        .copied()
        .zip(conditional.data().iter().copied())
        .map(|(uncond, cond)| uncond + (cond - uncond) * guidance_scale)
        .collect();
    SdTensor::new(unconditional.shape().to_vec(), data)
}

pub fn unet_conv2d(input: &SdTensor, weights: &UnetConv2dWeights) -> Result<SdTensor> {
    conv2d_nchw(
        input,
        &weights.weight,
        weights.bias.as_deref(),
        Conv2dOptions {
            stride: weights.stride,
            padding: weights.padding,
        },
    )
}

pub fn unet_linear(input: &SdTensor, weights: &UnetLinearWeights) -> Result<SdTensor> {
    if input.rank() != 2 || input.shape()[1] != weights.in_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet linear expected input [rows, {}], got {:?}",
            weights.in_features,
            input.shape()
        )));
    }
    if weights.weight.len() != weights.in_features * weights.out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet linear weight expected {} values, got {}",
            weights.in_features * weights.out_features,
            weights.weight.len()
        )));
    }
    if weights.bias.len() != weights.out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet linear bias expected {} values, got {}",
            weights.out_features,
            weights.bias.len()
        )));
    }
    linear2d(
        input,
        &weights.weight,
        Some(&weights.bias),
        weights.in_features,
        weights.out_features,
    )
}

fn unet_linear_flattened(input: &SdTensor, weights: &UnetLinearWeights) -> Result<SdTensor> {
    if input.rank() < 2 || input.shape()[input.rank() - 1] != weights.in_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet flattened linear expected trailing dim {}, got {:?}",
            weights.in_features,
            input.shape()
        )));
    }
    if weights.weight.len() != weights.in_features * weights.out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet linear weight expected {} values, got {}",
            weights.in_features * weights.out_features,
            weights.weight.len()
        )));
    }
    if weights.bias.len() != weights.out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet linear bias expected {} values, got {}",
            weights.out_features,
            weights.bias.len()
        )));
    }
    linear_flattened_last_dim(
        input,
        &weights.weight,
        Some(&weights.bias),
        weights.in_features,
        weights.out_features,
    )
}

pub fn unet_resnet_block(
    input: &SdTensor,
    time_embedding: &SdTensor,
    weights: &UnetResnetBlockWeights,
    groups: usize,
    eps: f32,
) -> Result<SdTensor> {
    let hidden = group_norm_silu_nchw(
        input,
        groups,
        &weights.norm1_weight,
        &weights.norm1_bias,
        eps,
    )?;
    let mut hidden = unet_conv2d(&hidden, &weights.conv1)?;
    let time = unet_linear(&time_embedding.silu()?, &weights.time_emb_proj)?;
    add_channel_bias_from_row(&mut hidden, &time)?;
    hidden = group_norm_silu_nchw(
        &hidden,
        groups,
        &weights.norm2_weight,
        &weights.norm2_bias,
        eps,
    )?;
    hidden = unet_conv2d(&hidden, &weights.conv2)?;
    if weights.output_scale_factor <= 0.0 || !weights.output_scale_factor.is_finite() {
        return Err(StableDiffusionError::InvalidInput(
            "UNet resnet output_scale_factor must be finite and > 0".to_string(),
        ));
    }
    match &weights.shortcut {
        Some(shortcut) => {
            let residual = unet_conv2d(input, shortcut)?;
            hidden.add_same_shape_in_place(&residual)?;
        }
        None => hidden.add_same_shape_in_place(input)?,
    }
    if weights.output_scale_factor == 1.0 {
        Ok(hidden)
    } else {
        hidden.scale(1.0 / weights.output_scale_factor)
    }
}

pub fn unet_attention(
    query_states: &SdTensor,
    encoder_hidden_states: &SdTensor,
    weights: &UnetAttentionWeights,
) -> Result<SdTensor> {
    let [query_batch, query_len, _query_dim] = shape3(query_states, "UNet attention query states")?;
    let [context_batch, _key_len, _context_dim] =
        shape3(encoder_hidden_states, "UNet attention encoder states")?;
    if query_batch != 1 || context_batch != 1 {
        return Err(StableDiffusionError::Unsupported(
            "native UNet attention currently supports batch size 1".to_string(),
        ));
    }
    if weights.heads == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "UNet attention heads must be > 0".to_string(),
        ));
    }

    let q = split_attention_heads(
        &unet_linear_flattened(query_states, &weights.to_q)?,
        weights.heads,
    )?;
    let k = split_attention_heads(
        &unet_linear_flattened(encoder_hidden_states, &weights.to_k)?,
        weights.heads,
    )?;
    let v = split_attention_heads(
        &unet_linear_flattened(encoder_hidden_states, &weights.to_v)?,
        weights.heads,
    )?;
    let attended = scaled_dot_product_attention(&q, &k, &v, None)?;
    let attended = merge_attention_heads(&attended)?;
    let out = unet_linear(&attended, &weights.to_out)?;
    out.into_shape([1, query_len, weights.to_out.out_features])
}

pub fn unet_feed_forward(input: &SdTensor, weights: &UnetFeedForwardWeights) -> Result<SdTensor> {
    let [batch, seq_len, _hidden] = shape3(input, "UNet feed-forward input")?;
    if batch != 1 {
        return Err(StableDiffusionError::Unsupported(
            "native UNet feed-forward currently supports batch size 1".to_string(),
        ));
    }
    let projected = unet_linear_flattened(input, &weights.geglu_proj)?;
    if projected.shape()[1] % 2 != 0 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet GEGLU projection output width {} must be even",
            projected.shape()[1]
        )));
    }
    let inner = projected.shape()[1] / 2;
    let mut gated = vec![0.0; seq_len * inner];
    for row in 0..seq_len {
        for col in 0..inner {
            let value = projected.data()[row * inner * 2 + col];
            let gate = projected.data()[row * inner * 2 + inner + col];
            gated[row * inner + col] = value * gelu_scalar(gate);
        }
    }
    let gated = SdTensor::new([seq_len, inner], gated)?;
    let out = unet_linear(&gated, &weights.out_proj)?;
    out.into_shape([1, seq_len, weights.out_proj.out_features])
}

pub fn unet_transformer_block(
    input: &SdTensor,
    encoder_hidden_states: &SdTensor,
    weights: &UnetTransformerBlockWeights,
    eps: f32,
) -> Result<SdTensor> {
    let norm1 = layer_norm_last_dim(input, &weights.norm1_weight, &weights.norm1_bias, eps)?;
    let _ = (&norm1, &weights.self_attn);
    let hidden = input.clone();

    let norm2 = layer_norm_last_dim(&hidden, &weights.norm2_weight, &weights.norm2_bias, eps)?;
    let cross_attn = unet_attention(&norm2, encoder_hidden_states, &weights.cross_attn)?;
    let hidden = hidden.add(&cross_attn)?;

    let norm3 = layer_norm_last_dim(&hidden, &weights.norm3_weight, &weights.norm3_bias, eps)?;
    let feed_forward = unet_feed_forward(&norm3, &weights.feed_forward)?;
    hidden.add(&feed_forward)
}

pub fn unet_spatial_transformer(
    input: &SdTensor,
    encoder_hidden_states: &SdTensor,
    weights: &UnetSpatialTransformerWeights,
    groups: usize,
    eps: f32,
) -> Result<SdTensor> {
    let [batch, channels, height, width] = nchw_shape(input, "UNet spatial transformer input")?;
    if batch != 1 {
        return Err(StableDiffusionError::Unsupported(
            "native UNet spatial transformer currently supports batch size 1".to_string(),
        ));
    }
    let residual = input.clone();
    let hidden = group_norm_nchw(input, groups, &weights.norm_weight, &weights.norm_bias, eps)?;
    let hidden = unet_conv2d(&hidden, &weights.proj_in)?;
    let inner_channels = hidden.shape()[1];
    let mut hidden = spatial_nchw_to_sequence(&hidden)?;
    for block in &weights.transformer_blocks {
        hidden = unet_transformer_block(&hidden, encoder_hidden_states, block, eps)?;
    }
    let hidden = sequence_to_spatial_nchw(&hidden, inner_channels, height, width)?;
    let hidden = unet_conv2d(&hidden, &weights.proj_out)?;
    if hidden.shape() != [batch, channels, height, width] {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet spatial transformer projected shape {:?} does not match residual shape {:?}",
            hidden.shape(),
            residual.shape()
        )));
    }
    residual.add(&hidden)
}

pub fn unet_down_block(
    input: &SdTensor,
    time_embedding: &SdTensor,
    encoder_hidden_states: &SdTensor,
    weights: &UnetDownBlockWeights,
    groups: usize,
    eps: f32,
) -> Result<(SdTensor, Vec<SdTensor>)> {
    if weights.resnets.len() != weights.attentions.len() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet down block has {} resnets but {} attention entries",
            weights.resnets.len(),
            weights.attentions.len()
        )));
    }
    let mut hidden = input.clone();
    let mut residuals =
        Vec::with_capacity(weights.resnets.len() + usize::from(weights.downsample.is_some()));
    for (resnet, attention) in weights.resnets.iter().zip(weights.attentions.iter()) {
        hidden = unet_resnet_block(&hidden, time_embedding, resnet, groups, eps)?;
        if let Some(attention) = attention {
            hidden =
                unet_spatial_transformer(&hidden, encoder_hidden_states, attention, groups, eps)?;
        }
        residuals.push(hidden.clone());
    }
    if let Some(downsample) = &weights.downsample {
        hidden = unet_conv2d(&hidden, downsample)?;
        residuals.push(hidden.clone());
    }
    Ok((hidden, residuals))
}

pub fn unet_mid_block(
    input: &SdTensor,
    time_embedding: &SdTensor,
    encoder_hidden_states: &SdTensor,
    weights: &UnetMidBlockWeights,
    groups: usize,
    eps: f32,
) -> Result<SdTensor> {
    if weights.attentions.len() != weights.resnets.len() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet mid block has {} attentions but {} trailing resnets",
            weights.attentions.len(),
            weights.resnets.len()
        )));
    }
    let mut hidden = unet_resnet_block(input, time_embedding, &weights.resnet1, groups, eps)?;
    for (attention, resnet) in weights.attentions.iter().zip(weights.resnets.iter()) {
        hidden = unet_spatial_transformer(&hidden, encoder_hidden_states, attention, groups, eps)?;
        hidden = unet_resnet_block(&hidden, time_embedding, resnet, groups, eps)?;
    }
    Ok(hidden)
}

pub fn unet_up_block(
    input: &SdTensor,
    residuals: &mut Vec<SdTensor>,
    time_embedding: &SdTensor,
    encoder_hidden_states: &SdTensor,
    weights: &UnetUpBlockWeights,
    groups: usize,
    eps: f32,
) -> Result<SdTensor> {
    if weights.resnets.len() != weights.attentions.len() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet up block has {} resnets but {} attention entries",
            weights.resnets.len(),
            weights.attentions.len()
        )));
    }
    let mut hidden = input.clone();
    for (resnet, attention) in weights.resnets.iter().zip(weights.attentions.iter()) {
        let skip = residuals.pop().ok_or_else(|| {
            StableDiffusionError::InvalidInput(
                "UNet up block requested more residual skips than were available".to_string(),
            )
        })?;
        hidden = concat_tensors(1, &[hidden, skip])?;
        hidden = unet_resnet_block(&hidden, time_embedding, resnet, groups, eps)?;
        if let Some(attention) = attention {
            hidden =
                unet_spatial_transformer(&hidden, encoder_hidden_states, attention, groups, eps)?;
        }
    }
    if let Some(upsample) = &weights.upsample {
        hidden = upsample_nearest2d_nchw(&hidden, 2)?;
        hidden = unet_conv2d(&hidden, upsample)?;
    }
    Ok(hidden)
}

pub fn unet_forward(
    sample: &SdTensor,
    timestep: usize,
    encoder_hidden_states: &SdTensor,
    weights: &Unet2DConditionModelWeights,
) -> Result<SdTensor> {
    let mut hidden = unet_conv2d(sample, &weights.conv_in)?;
    let mut time = timestep_embedding(
        timestep,
        weights.time_embedding_linear1.in_features,
        10_000.0,
    )?;
    time = unet_linear(&time, &weights.time_embedding_linear1)?.silu()?;
    time = unet_linear(&time, &weights.time_embedding_linear2)?;

    let mut residuals = vec![hidden.clone()];
    for down_block in &weights.down_blocks {
        let (next, mut block_residuals) = unet_down_block(
            &hidden,
            &time,
            encoder_hidden_states,
            down_block,
            weights.norm_groups,
            weights.norm_eps,
        )?;
        residuals.append(&mut block_residuals);
        hidden = next;
    }

    hidden = unet_mid_block(
        &hidden,
        &time,
        encoder_hidden_states,
        &weights.mid_block,
        weights.norm_groups,
        weights.norm_eps,
    )?;

    for up_block in &weights.up_blocks {
        hidden = unet_up_block(
            &hidden,
            &mut residuals,
            &time,
            encoder_hidden_states,
            up_block,
            weights.norm_groups,
            weights.norm_eps,
        )?;
    }

    let hidden = group_norm_silu_nchw(
        &hidden,
        weights.norm_groups,
        &weights.conv_norm_out_weight,
        &weights.conv_norm_out_bias,
        weights.norm_eps,
    )?;
    let out = unet_conv2d(&hidden, &weights.conv_out)?;
    if out.shape() != sample.shape() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet output shape {:?} does not match input sample shape {:?}",
            out.shape(),
            sample.shape()
        )));
    }
    Ok(out)
}

pub fn spatial_nchw_to_sequence(input: &SdTensor) -> Result<SdTensor> {
    let [batch, channels, height, width] = nchw_shape(input, "UNet spatial to sequence")?;
    if batch != 1 {
        return Err(StableDiffusionError::Unsupported(
            "native UNet spatial attention currently supports batch size 1".to_string(),
        ));
    }
    let tokens = height * width;
    let mut out = vec![0.0; tokens * channels];
    for y in 0..height {
        for x in 0..width {
            let token = y * width + x;
            for channel in 0..channels {
                out[token * channels + channel] = input.data()[(channel * height + y) * width + x];
            }
        }
    }
    SdTensor::new([1, tokens, channels], out)
}

pub fn sequence_to_spatial_nchw(
    input: &SdTensor,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<SdTensor> {
    if input.shape() != [1, height * width, channels] {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet sequence shape {:?} does not match expected {:?}",
            input.shape(),
            [1, height * width, channels]
        )));
    }
    let mut out = vec![0.0; channels * height * width];
    for y in 0..height {
        for x in 0..width {
            let token = y * width + x;
            for channel in 0..channels {
                out[(channel * height + y) * width + x] = input.data()[token * channels + channel];
            }
        }
    }
    SdTensor::new([1, channels, height, width], out)
}

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn add_channel_bias_from_row(input: &mut SdTensor, row: &SdTensor) -> Result<()> {
    if row.rank() != 2 || row.shape()[0] != 1 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet time embedding projection must be [1, channels], got {:?}",
            row.shape()
        )));
    }
    let [batch, channels, height, width] = nchw_shape(input, "UNet time add")?;
    if row.shape()[1] != channels {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet time embedding projection channels {} do not match hidden channels {channels}",
            row.shape()[1]
        )));
    }
    for b in 0..batch {
        for channel in 0..channels {
            let bias = row.data()[channel];
            for y in 0..height {
                for x in 0..width {
                    input.data_mut()[((b * channels + channel) * height + y) * width + x] += bias;
                }
            }
        }
    }
    Ok(())
}

fn split_attention_heads(input: &SdTensor, heads: usize) -> Result<SdTensor> {
    if input.rank() != 2 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet split attention heads requires rank-2 input, got {:?}",
            input.shape()
        )));
    }
    let seq_len = input.shape()[0];
    let hidden = input.shape()[1];
    if hidden % heads != 0 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet attention hidden size {hidden} must be divisible by {heads} heads"
        )));
    }
    let head_dim = hidden / heads;
    let mut out = vec![0.0; heads * seq_len * head_dim];
    for seq in 0..seq_len {
        for head in 0..heads {
            for dim in 0..head_dim {
                out[(head * seq_len + seq) * head_dim + dim] =
                    input.data()[seq * hidden + head * head_dim + dim];
            }
        }
    }
    SdTensor::new([1, heads, seq_len, head_dim], out)
}

fn merge_attention_heads(input: &SdTensor) -> Result<SdTensor> {
    if input.rank() != 4 || input.shape()[0] != 1 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet merge attention heads expects [1, heads, seq, dim], got {:?}",
            input.shape()
        )));
    }
    let heads = input.shape()[1];
    let seq_len = input.shape()[2];
    let head_dim = input.shape()[3];
    let hidden = heads * head_dim;
    let mut out = vec![0.0; seq_len * hidden];
    for seq in 0..seq_len {
        for head in 0..heads {
            for dim in 0..head_dim {
                out[seq * hidden + head * head_dim + dim] =
                    input.data()[(head * seq_len + seq) * head_dim + dim];
            }
        }
    }
    SdTensor::new([seq_len, hidden], out)
}

pub fn validate_unet_latent_shape(latents: &SdTensor, width: u32, height: u32) -> Result<()> {
    let expected = [1, 4, (height / 8) as usize, (width / 8) as usize];
    if latents.shape() != expected {
        return Err(StableDiffusionError::InvalidInput(format!(
            "UNet latent shape {:?} does not match expected {:?}",
            latents.shape(),
            expected
        )));
    }
    Ok(())
}

fn shape3(tensor: &SdTensor, name: &str) -> Result<[usize; 3]> {
    if tensor.rank() != 3 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "{name} requires rank-3 tensor, got {:?}",
            tensor.shape()
        )));
    }
    Ok([tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]])
}

fn nchw_shape(tensor: &SdTensor, name: &str) -> Result<[usize; 4]> {
    if tensor.rank() != 4 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "{name} requires rank-4 NCHW tensor, got {:?}",
            tensor.shape()
        )));
    }
    Ok([
        tensor.shape()[0],
        tensor.shape()[1],
        tensor.shape()[2],
        tensor.shape()[3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_diffusers_style_timestep_embedding() -> Result<()> {
        let embedding = timestep_embedding(10, 5, 10_000.0)?;

        assert_eq!(embedding.shape(), &[1, 5]);
        assert!((embedding.data()[0] - 10.0f32.cos()).abs() < 1e-6);
        assert!((embedding.data()[2] - 10.0f32.sin()).abs() < 1e-6);
        assert_eq!(embedding.data()[4], 0.0);
        Ok(())
    }

    #[test]
    fn combines_classifier_free_guidance_predictions() -> Result<()> {
        let uncond = SdTensor::new([1, 2], vec![1.0, 3.0])?;
        let cond = SdTensor::new([1, 2], vec![2.0, 1.0])?;

        let guided = classifier_free_guidance(&uncond, &cond, 1.5)?;

        assert_eq!(guided.data(), &[2.5, 0.0]);
        Ok(())
    }

    #[test]
    fn validates_latent_shape() -> Result<()> {
        let latents = SdTensor::zeros([1, 4, 64, 64])?;

        validate_unet_latent_shape(&latents, 512, 512)?;
        assert!(validate_unet_latent_shape(&latents, 512, 256).is_err());
        Ok(())
    }

    fn pointwise_conv(channels: usize, diagonal: f32) -> UnetConv2dWeights {
        let mut data = vec![0.0; channels * channels];
        for channel in 0..channels {
            data[channel * channels + channel] = diagonal;
        }
        UnetConv2dWeights {
            weight: SdTensor::new([channels, channels, 1, 1], data).unwrap(),
            bias: Some(vec![0.0; channels]),
            stride: 1,
            padding: 0,
        }
    }

    #[test]
    fn unet_linear_applies_transposed_diffusers_weight_layout() -> Result<()> {
        let input = SdTensor::new([1, 2], vec![2.0, 3.0])?;
        let weights = UnetLinearWeights {
            weight: vec![1.0, 3.0, 2.0, 4.0],
            bias: vec![0.5, -0.5],
            in_features: 2,
            out_features: 2,
        };

        let out = unet_linear(&input, &weights)?;

        assert_eq!(out.data(), &[8.5, 17.5]);
        Ok(())
    }

    #[test]
    fn unet_resnet_block_runs_timestep_projection_and_residual() -> Result<()> {
        let input = SdTensor::new([1, 2, 1, 2], vec![1.0, -1.0, 0.5, -0.5])?;
        let time_embedding = SdTensor::new([1, 2], vec![0.0, 0.0])?;
        let block = UnetResnetBlockWeights {
            norm1_weight: vec![1.0, 1.0],
            norm1_bias: vec![0.0, 0.0],
            conv1: pointwise_conv(2, 0.0),
            time_emb_proj: UnetLinearWeights {
                weight: vec![0.0; 4],
                bias: vec![0.0, 0.0],
                in_features: 2,
                out_features: 2,
            },
            norm2_weight: vec![1.0, 1.0],
            norm2_bias: vec![0.0, 0.0],
            conv2: pointwise_conv(2, 0.0),
            shortcut: None,
            output_scale_factor: 1.0,
        };

        let out = unet_resnet_block(&input, &time_embedding, &block, 2, 1e-5)?;

        assert_eq!(out, input);
        Ok(())
    }

    #[test]
    fn unet_conv2d_supports_stride_for_downsampling() -> Result<()> {
        let input = SdTensor::new([1, 1, 4, 4], (1..=16).map(|value| value as f32).collect())?;
        let weights = UnetConv2dWeights {
            weight: SdTensor::new([1, 1, 1, 1], vec![1.0])?,
            bias: Some(vec![0.0]),
            stride: 2,
            padding: 0,
        };

        let out = unet_conv2d(&input, &weights)?;

        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.data(), &[1.0, 3.0, 9.0, 11.0]);
        Ok(())
    }

    fn linear_identity(features: usize) -> UnetLinearWeights {
        let mut weight = vec![0.0; features * features];
        for feature in 0..features {
            weight[feature * features + feature] = 1.0;
        }
        UnetLinearWeights {
            weight,
            bias: vec![0.0; features],
            in_features: features,
            out_features: features,
        }
    }

    #[test]
    fn unet_attention_supports_cross_attention_context() -> Result<()> {
        let query = SdTensor::new([1, 2, 1], vec![0.0, 0.0])?;
        let context = SdTensor::new([1, 2, 1], vec![1.0, 3.0])?;
        let weights = UnetAttentionWeights {
            to_q: linear_identity(1),
            to_k: UnetLinearWeights {
                weight: vec![0.0],
                bias: vec![0.0],
                in_features: 1,
                out_features: 1,
            },
            to_v: linear_identity(1),
            to_out: linear_identity(1),
            heads: 1,
        };

        let out = unet_attention(&query, &context, &weights)?;

        assert_eq!(out.shape(), &[1, 2, 1]);
        assert!((out.data()[0] - 2.0).abs() < 1e-6);
        assert!((out.data()[1] - 2.0).abs() < 1e-6);
        Ok(())
    }

    fn zero_attention(features: usize) -> UnetAttentionWeights {
        let zero = UnetLinearWeights {
            weight: vec![0.0; features * features],
            bias: vec![0.0; features],
            in_features: features,
            out_features: features,
        };
        UnetAttentionWeights {
            to_q: zero.clone(),
            to_k: zero.clone(),
            to_v: zero.clone(),
            to_out: zero,
            heads: 1,
        }
    }

    #[test]
    fn unet_feed_forward_runs_geglu_projection() -> Result<()> {
        let input = SdTensor::new([1, 1, 1], vec![2.0])?;
        let weights = UnetFeedForwardWeights {
            geglu_proj: UnetLinearWeights {
                weight: vec![1.0, 1.0],
                bias: vec![0.0, 0.0],
                in_features: 1,
                out_features: 2,
            },
            out_proj: linear_identity(1),
        };

        let out = unet_feed_forward(&input, &weights)?;

        assert_eq!(out.shape(), &[1, 1, 1]);
        assert!(out.data()[0] > 3.9 && out.data()[0] < 4.0);
        Ok(())
    }

    #[test]
    fn unet_transformer_block_preserves_shape_with_zero_sublayers() -> Result<()> {
        let input = SdTensor::new([1, 2, 2], vec![1.0, -1.0, -0.5, 0.5])?;
        let context = SdTensor::new([1, 2, 2], vec![0.25, -0.25, 0.5, -0.5])?;
        let zero_ff = UnetFeedForwardWeights {
            geglu_proj: UnetLinearWeights {
                weight: vec![0.0; 8],
                bias: vec![0.0; 4],
                in_features: 2,
                out_features: 4,
            },
            out_proj: UnetLinearWeights {
                weight: vec![0.0; 4],
                bias: vec![0.0; 2],
                in_features: 2,
                out_features: 2,
            },
        };
        let block = UnetTransformerBlockWeights {
            norm1_weight: vec![1.0, 1.0],
            norm1_bias: vec![0.0, 0.0],
            self_attn: zero_attention(2),
            norm2_weight: vec![1.0, 1.0],
            norm2_bias: vec![0.0, 0.0],
            cross_attn: zero_attention(2),
            norm3_weight: vec![1.0, 1.0],
            norm3_bias: vec![0.0, 0.0],
            feed_forward: zero_ff,
        };

        let out = unet_transformer_block(&input, &context, &block, 1e-5)?;

        assert_eq!(out, input);
        Ok(())
    }

    #[test]
    fn unet_spatial_transformer_preserves_shape_with_zero_projection() -> Result<()> {
        let input = SdTensor::new([1, 2, 1, 2], vec![1.0, -1.0, 0.5, -0.5])?;
        let context = SdTensor::new([1, 2, 2], vec![0.25, -0.25, 0.5, -0.5])?;
        let zero_ff = UnetFeedForwardWeights {
            geglu_proj: UnetLinearWeights {
                weight: vec![0.0; 8],
                bias: vec![0.0; 4],
                in_features: 2,
                out_features: 4,
            },
            out_proj: UnetLinearWeights {
                weight: vec![0.0; 4],
                bias: vec![0.0; 2],
                in_features: 2,
                out_features: 2,
            },
        };
        let block = UnetTransformerBlockWeights {
            norm1_weight: vec![1.0, 1.0],
            norm1_bias: vec![0.0, 0.0],
            self_attn: zero_attention(2),
            norm2_weight: vec![1.0, 1.0],
            norm2_bias: vec![0.0, 0.0],
            cross_attn: zero_attention(2),
            norm3_weight: vec![1.0, 1.0],
            norm3_bias: vec![0.0, 0.0],
            feed_forward: zero_ff,
        };
        let weights = UnetSpatialTransformerWeights {
            norm_weight: vec![1.0, 1.0],
            norm_bias: vec![0.0, 0.0],
            proj_in: pointwise_conv(2, 0.0),
            transformer_blocks: vec![block],
            proj_out: pointwise_conv(2, 0.0),
        };

        let out = unet_spatial_transformer(&input, &context, &weights, 2, 1e-5)?;

        assert_eq!(out, input);
        Ok(())
    }

    fn zero_resnet_block(channels: usize, time_dim: usize) -> UnetResnetBlockWeights {
        UnetResnetBlockWeights {
            norm1_weight: vec![1.0; channels],
            norm1_bias: vec![0.0; channels],
            conv1: pointwise_conv(channels, 0.0),
            time_emb_proj: UnetLinearWeights {
                weight: vec![0.0; time_dim * channels],
                bias: vec![0.0; channels],
                in_features: time_dim,
                out_features: channels,
            },
            norm2_weight: vec![1.0; channels],
            norm2_bias: vec![0.0; channels],
            conv2: pointwise_conv(channels, 0.0),
            shortcut: None,
            output_scale_factor: 1.0,
        }
    }

    fn zero_resnet_block_with_shortcut(
        in_channels: usize,
        out_channels: usize,
        time_dim: usize,
    ) -> UnetResnetBlockWeights {
        let conv1 = UnetConv2dWeights {
            weight: SdTensor::zeros([out_channels, in_channels, 1, 1]).unwrap(),
            bias: Some(vec![0.0; out_channels]),
            stride: 1,
            padding: 0,
        };
        let conv2 = UnetConv2dWeights {
            weight: SdTensor::zeros([out_channels, out_channels, 1, 1]).unwrap(),
            bias: Some(vec![0.0; out_channels]),
            stride: 1,
            padding: 0,
        };
        let mut shortcut_weight = vec![0.0; out_channels * in_channels];
        for channel in 0..out_channels.min(in_channels) {
            shortcut_weight[channel * in_channels + channel] = 1.0;
        }
        UnetResnetBlockWeights {
            norm1_weight: vec![1.0; in_channels],
            norm1_bias: vec![0.0; in_channels],
            conv1,
            time_emb_proj: UnetLinearWeights {
                weight: vec![0.0; time_dim * out_channels],
                bias: vec![0.0; out_channels],
                in_features: time_dim,
                out_features: out_channels,
            },
            norm2_weight: vec![1.0; out_channels],
            norm2_bias: vec![0.0; out_channels],
            conv2,
            shortcut: Some(UnetConv2dWeights {
                weight: SdTensor::new([out_channels, in_channels, 1, 1], shortcut_weight).unwrap(),
                bias: Some(vec![0.0; out_channels]),
                stride: 1,
                padding: 0,
            }),
            output_scale_factor: 1.0,
        }
    }

    #[test]
    fn unet_down_block_collects_residuals_and_downsamples() -> Result<()> {
        let input = SdTensor::new([1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
        let time = SdTensor::new([1, 1], vec![0.0])?;
        let context = SdTensor::new([1, 1, 1], vec![0.0])?;
        let downsample = UnetConv2dWeights {
            weight: SdTensor::new([1, 1, 1, 1], vec![1.0])?,
            bias: Some(vec![0.0]),
            stride: 2,
            padding: 0,
        };
        let block = UnetDownBlockWeights {
            resnets: vec![zero_resnet_block(1, 1)],
            attentions: vec![None],
            downsample: Some(downsample),
        };

        let (hidden, residuals) = unet_down_block(&input, &time, &context, &block, 1, 1e-5)?;

        assert_eq!(residuals.len(), 2);
        assert_eq!(residuals[0], input);
        assert_eq!(hidden.shape(), &[1, 1, 1, 1]);
        assert_eq!(hidden.data(), &[1.0]);
        Ok(())
    }

    #[test]
    fn unet_mid_block_runs_resnet_attention_resnet_sequence() -> Result<()> {
        let input = SdTensor::new([1, 1, 1, 2], vec![1.0, -1.0])?;
        let time = SdTensor::new([1, 1], vec![0.0])?;
        let context = SdTensor::new([1, 2, 1], vec![0.0, 0.0])?;
        let attention = UnetSpatialTransformerWeights {
            norm_weight: vec![1.0],
            norm_bias: vec![0.0],
            proj_in: pointwise_conv(1, 0.0),
            transformer_blocks: Vec::new(),
            proj_out: pointwise_conv(1, 0.0),
        };
        let block = UnetMidBlockWeights {
            resnet1: zero_resnet_block(1, 1),
            attentions: vec![attention],
            resnets: vec![zero_resnet_block(1, 1)],
        };

        let out = unet_mid_block(&input, &time, &context, &block, 1, 1e-5)?;

        assert_eq!(out, input);
        Ok(())
    }

    #[test]
    fn unet_up_block_consumes_skips_and_upsamples() -> Result<()> {
        let input = SdTensor::new([1, 1, 1, 1], vec![2.0])?;
        let mut residuals = vec![SdTensor::new([1, 1, 1, 1], vec![10.0])?];
        let time = SdTensor::new([1, 1], vec![0.0])?;
        let context = SdTensor::new([1, 1, 1], vec![0.0])?;
        let upsample = UnetConv2dWeights {
            weight: SdTensor::new([1, 1, 1, 1], vec![1.0])?,
            bias: Some(vec![0.0]),
            stride: 1,
            padding: 0,
        };
        let block = UnetUpBlockWeights {
            resnets: vec![zero_resnet_block_with_shortcut(2, 1, 1)],
            attentions: vec![None],
            upsample: Some(upsample),
        };

        let out = unet_up_block(&input, &mut residuals, &time, &context, &block, 1, 1e-5)?;

        assert!(residuals.is_empty());
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.data(), &[2.0, 2.0, 2.0, 2.0]);
        Ok(())
    }

    #[test]
    fn unet_forward_runs_tiny_zero_network() -> Result<()> {
        let sample = SdTensor::new([1, 1, 1, 2], vec![1.0, -1.0])?;
        let context = SdTensor::new([1, 2, 1], vec![0.0, 0.0])?;
        let weights = Unet2DConditionModelWeights {
            conv_in: pointwise_conv(1, 1.0),
            time_embedding_linear1: UnetLinearWeights {
                weight: vec![0.0],
                bias: vec![0.0],
                in_features: 1,
                out_features: 1,
            },
            time_embedding_linear2: UnetLinearWeights {
                weight: vec![0.0],
                bias: vec![0.0],
                in_features: 1,
                out_features: 1,
            },
            down_blocks: Vec::new(),
            mid_block: UnetMidBlockWeights {
                resnet1: zero_resnet_block(1, 1),
                attentions: Vec::new(),
                resnets: Vec::new(),
            },
            up_blocks: vec![UnetUpBlockWeights {
                resnets: vec![zero_resnet_block_with_shortcut(2, 1, 1)],
                attentions: vec![None],
                upsample: None,
            }],
            conv_norm_out_weight: vec![1.0],
            conv_norm_out_bias: vec![0.0],
            conv_out: pointwise_conv(1, 0.0),
            time_embedding_dim: 1,
            norm_groups: 1,
            norm_eps: 1e-5,
        };

        let out = unet_forward(&sample, 0, &context, &weights)?;

        assert_eq!(out.shape(), sample.shape());
        assert!(out.data().iter().all(|value| value.abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn spatial_sequence_roundtrip_preserves_nchw_values() -> Result<()> {
        let input = SdTensor::new([1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0])?;

        let seq = spatial_nchw_to_sequence(&input)?;
        let out = sequence_to_spatial_nchw(&seq, 2, 1, 2)?;

        assert_eq!(seq.shape(), &[1, 2, 2]);
        assert_eq!(seq.data(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(out, input);
        Ok(())
    }
}
