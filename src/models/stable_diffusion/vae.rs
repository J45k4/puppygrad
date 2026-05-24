use super::{
    conv2d_nchw, group_norm_nchw, group_norm_silu_nchw, scaled_dot_product_attention,
    upsample_nearest2d_nchw, AutoencoderKlConfig, Conv2dOptions, Result, SdTensor,
    StableDiffusionError,
};

#[derive(Clone, Debug, PartialEq)]
pub struct VaeConv2dWeights {
    pub weight: SdTensor,
    pub bias: Option<Vec<f32>>,
    pub padding: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaeResnetBlockWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub conv1: VaeConv2dWeights,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub conv2: VaeConv2dWeights,
    pub shortcut: Option<VaeConv2dWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaeAttentionBlockWeights {
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub query: VaeConv2dWeights,
    pub key: VaeConv2dWeights,
    pub value: VaeConv2dWeights,
    pub proj_attn: VaeConv2dWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaeMidBlockWeights {
    pub resnet1: VaeResnetBlockWeights,
    pub attention: Option<VaeAttentionBlockWeights>,
    pub resnet2: VaeResnetBlockWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaeUpDecoderBlockWeights {
    pub resnets: Vec<VaeResnetBlockWeights>,
    pub upsample: Option<VaeConv2dWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaeDecoderModelWeights {
    pub post_quant_conv: Option<VaeConv2dWeights>,
    pub conv_in: VaeConv2dWeights,
    pub mid_block: VaeMidBlockWeights,
    pub up_blocks: Vec<VaeUpDecoderBlockWeights>,
    pub conv_norm_out_weight: Vec<f32>,
    pub conv_norm_out_bias: Vec<f32>,
    pub conv_out: VaeConv2dWeights,
}

pub fn vae_conv2d(input: &SdTensor, weights: &VaeConv2dWeights) -> Result<SdTensor> {
    conv2d_nchw(
        input,
        &weights.weight,
        weights.bias.as_deref(),
        Conv2dOptions {
            stride: 1,
            padding: weights.padding,
        },
    )
}

pub fn vae_resnet_block(
    input: &SdTensor,
    weights: &VaeResnetBlockWeights,
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
    let hidden = vae_conv2d(&hidden, &weights.conv1)?;
    let hidden = group_norm_silu_nchw(
        &hidden,
        groups,
        &weights.norm2_weight,
        &weights.norm2_bias,
        eps,
    )?;
    let mut hidden = vae_conv2d(&hidden, &weights.conv2)?;
    match &weights.shortcut {
        Some(shortcut) => {
            let residual = vae_conv2d(input, shortcut)?;
            hidden.add_same_shape_in_place(&residual)?;
        }
        None => hidden.add_same_shape_in_place(input)?,
    }
    Ok(hidden)
}

pub fn vae_attention_block(
    input: &SdTensor,
    weights: &VaeAttentionBlockWeights,
    groups: usize,
    eps: f32,
) -> Result<SdTensor> {
    let [batch, channels, height, width] = nchw_shape(input, "VAE attention input")?;
    if batch != 1 {
        return Err(StableDiffusionError::Unsupported(
            "native VAE attention currently supports batch size 1".to_string(),
        ));
    }
    let norm = group_norm_nchw(input, groups, &weights.norm_weight, &weights.norm_bias, eps)?;
    let query = spatial_conv_to_attention(&vae_conv2d(&norm, &weights.query)?)?;
    let key = spatial_conv_to_attention(&vae_conv2d(&norm, &weights.key)?)?;
    let value = spatial_conv_to_attention(&vae_conv2d(&norm, &weights.value)?)?;
    let attended = scaled_dot_product_attention(&query, &key, &value, None)?;
    let attended = attention_to_spatial(&attended, channels, height, width)?;
    let projected = vae_conv2d(&attended, &weights.proj_attn)?;
    input.add(&projected)
}

pub fn vae_decode_latents(
    latents: &SdTensor,
    config: &AutoencoderKlConfig,
    weights: &VaeDecoderModelWeights,
) -> Result<SdTensor> {
    let norm_groups = config.norm_num_groups.ok_or_else(|| {
        StableDiffusionError::Config(
            "native VAE decoder requires norm_num_groups in vae/config.json".to_string(),
        )
    })?;
    let mut hidden = latents.scale(1.0 / config.scaling_factor)?;
    if let Some(post_quant_conv) = &weights.post_quant_conv {
        hidden = vae_conv2d(&hidden, post_quant_conv)?;
    }
    hidden = vae_conv2d(&hidden, &weights.conv_in)?;
    hidden = vae_mid_block(&hidden, &weights.mid_block, norm_groups, 1e-6)?;
    for up_block in &weights.up_blocks {
        for resnet in &up_block.resnets {
            hidden = vae_resnet_block(&hidden, resnet, norm_groups, 1e-6)?;
        }
        if let Some(upsample) = &up_block.upsample {
            hidden = upsample_nearest2d_nchw(&hidden, 2)?;
            hidden = vae_conv2d(&hidden, upsample)?;
        }
    }
    let hidden = group_norm_silu_nchw(
        &hidden,
        norm_groups,
        &weights.conv_norm_out_weight,
        &weights.conv_norm_out_bias,
        1e-6,
    )?;
    vae_conv2d(&hidden, &weights.conv_out)
}

fn vae_mid_block(
    input: &SdTensor,
    weights: &VaeMidBlockWeights,
    groups: usize,
    eps: f32,
) -> Result<SdTensor> {
    let mut hidden = vae_resnet_block(input, &weights.resnet1, groups, eps)?;
    if let Some(attention) = &weights.attention {
        hidden = vae_attention_block(&hidden, attention, groups, eps)?;
    }
    vae_resnet_block(&hidden, &weights.resnet2, groups, eps)
}

fn spatial_conv_to_attention(input: &SdTensor) -> Result<SdTensor> {
    let [batch, channels, height, width] = nchw_shape(input, "VAE attention projection")?;
    let tokens = height * width;
    let mut out = vec![0.0; batch * tokens * channels];
    for b in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let token = y * width + x;
                for channel in 0..channels {
                    out[(b * tokens + token) * channels + channel] =
                        input.data()[((b * channels + channel) * height + y) * width + x];
                }
            }
        }
    }
    SdTensor::new([batch, 1, tokens, channels], out)
}

fn attention_to_spatial(
    input: &SdTensor,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<SdTensor> {
    if input.shape() != [1, 1, height * width, channels] {
        return Err(StableDiffusionError::InvalidInput(format!(
            "VAE attention output shape {:?} does not match expected {:?}",
            input.shape(),
            [1, 1, height * width, channels]
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

pub fn validate_decoded_rgb_shape(image: &SdTensor, width: u32, height: u32) -> Result<()> {
    let expected = [1, 3, height as usize, width as usize];
    if image.shape() != expected {
        return Err(StableDiffusionError::InvalidInput(format!(
            "decoded RGB tensor shape {:?} does not match expected {:?}",
            image.shape(),
            expected
        )));
    }
    if !image.is_finite() {
        return Err(StableDiffusionError::InvalidInput(
            "decoded RGB tensor contains non-finite values".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pointwise_conv(channels: usize, diagonal: f32) -> VaeConv2dWeights {
        let mut data = vec![0.0; channels * channels];
        for channel in 0..channels {
            data[channel * channels + channel] = diagonal;
        }
        VaeConv2dWeights {
            weight: SdTensor::new([channels, channels, 1, 1], data).unwrap(),
            bias: Some(vec![0.0; channels]),
            padding: 0,
        }
    }

    fn identity_resnet_block(channels: usize) -> VaeResnetBlockWeights {
        let zero_conv = pointwise_conv(channels, 0.0);
        VaeResnetBlockWeights {
            norm1_weight: vec![1.0; channels],
            norm1_bias: vec![0.0; channels],
            conv1: zero_conv.clone(),
            norm2_weight: vec![1.0; channels],
            norm2_bias: vec![0.0; channels],
            conv2: zero_conv,
            shortcut: None,
        }
    }

    #[test]
    fn vae_resnet_block_preserves_shape_and_adds_residual() -> Result<()> {
        let input = SdTensor::new([1, 2, 1, 2], vec![1.0, -1.0, 0.5, -0.5])?;
        let zero_conv = pointwise_conv(2, 0.0);
        let block = VaeResnetBlockWeights {
            norm1_weight: vec![1.0, 1.0],
            norm1_bias: vec![0.0, 0.0],
            conv1: zero_conv.clone(),
            norm2_weight: vec![1.0, 1.0],
            norm2_bias: vec![0.0, 0.0],
            conv2: zero_conv,
            shortcut: None,
        };

        let out = vae_resnet_block(&input, &block, 2, 1e-6)?;

        assert_eq!(out, input);
        Ok(())
    }

    #[test]
    fn vae_attention_block_runs_spatial_self_attention() -> Result<()> {
        let input = SdTensor::new([1, 1, 1, 2], vec![1.0, 2.0])?;
        let block = VaeAttentionBlockWeights {
            norm_weight: vec![1.0],
            norm_bias: vec![0.0],
            query: pointwise_conv(1, 1.0),
            key: pointwise_conv(1, 1.0),
            value: pointwise_conv(1, 1.0),
            proj_attn: pointwise_conv(1, 0.0),
        };

        let out = vae_attention_block(&input, &block, 1, 1e-6)?;

        assert_eq!(out.shape(), &[1, 1, 1, 2]);
        assert_eq!(out, input);
        Ok(())
    }

    #[test]
    fn validates_decoded_rgb_shape_and_finiteness() -> Result<()> {
        let image = SdTensor::zeros([1, 3, 2, 2])?;

        validate_decoded_rgb_shape(&image, 2, 2)?;
        assert!(validate_decoded_rgb_shape(&image, 1, 2).is_err());
        Ok(())
    }

    #[test]
    fn vae_decoder_forward_composes_mid_and_up_blocks() -> Result<()> {
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
        let weights = VaeDecoderModelWeights {
            post_quant_conv: None,
            conv_in: pointwise_conv(1, 1.0),
            mid_block: VaeMidBlockWeights {
                resnet1: identity_resnet_block(1),
                attention: None,
                resnet2: identity_resnet_block(1),
            },
            up_blocks: vec![VaeUpDecoderBlockWeights {
                resnets: vec![identity_resnet_block(1)],
                upsample: None,
            }],
            conv_norm_out_weight: vec![1.0],
            conv_norm_out_bias: vec![0.0],
            conv_out: VaeConv2dWeights {
                weight: SdTensor::new([3, 1, 1, 1], vec![1.0, 0.5, -1.0])?,
                bias: Some(vec![0.0, 0.0, 0.0]),
                padding: 0,
            },
        };
        let latents = SdTensor::new([1, 1, 1, 2], vec![1.0, -1.0])?;

        let decoded = vae_decode_latents(&latents, &config, &weights)?;

        assert_eq!(decoded.shape(), &[1, 3, 1, 2]);
        assert!(decoded.is_finite());
        Ok(())
    }
}
