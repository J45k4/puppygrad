use super::config::ResNetConfig;
use crate::models::safetensors::{read_safetensors_file, SafeTensorLoadError, TensorStore};
use crate::vision::cnn::Conv2dShape;
use std::error;
use std::fmt;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct FoldedConv2d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub shape: Conv2dShape,
    pub stride: usize,
    pub padding: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlockWeights {
    pub conv1: FoldedConv2d,
    pub conv2: FoldedConv2d,
    pub downsample: Option<FoldedConv2d>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResNet18Weights {
    pub stem: FoldedConv2d,
    pub stages: Vec<Vec<BasicBlockWeights>>,
    pub fc_weight: Vec<f32>,
    pub fc_bias: Vec<f32>,
}

#[derive(Debug)]
pub enum ResNetWeightError {
    SafeTensor(SafeTensorLoadError),
    UnsupportedShape(String),
}

impl fmt::Display for ResNetWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResNetWeightError::SafeTensor(source) => write!(f, "{source}"),
            ResNetWeightError::UnsupportedShape(message) => write!(f, "{message}"),
        }
    }
}

impl error::Error for ResNetWeightError {}

impl From<SafeTensorLoadError> for ResNetWeightError {
    fn from(value: SafeTensorLoadError) -> Self {
        Self::SafeTensor(value)
    }
}

pub type Result<T> = std::result::Result<T, ResNetWeightError>;

pub fn load_resnet18_weights(path: &Path, config: &ResNetConfig) -> Result<ResNet18Weights> {
    let bytes = read_safetensors_file(path)?;
    let store = TensorStore::from_bytes(path, &bytes)?;
    load_resnet18_weights_from_store(&store, config)
}

pub fn load_resnet18_weights_from_store(
    store: &TensorStore<'_>,
    config: &ResNetConfig,
) -> Result<ResNet18Weights> {
    let stem = load_folded_conv(
        store,
        "conv1",
        "bn1",
        config.stage_channels[0],
        config.input_channels,
        config.stem_kernel,
        config.stem_stride,
        config.stem_padding,
        config.batch_norm_epsilon,
    )?;

    let mut stages = Vec::with_capacity(config.stage_channels.len());
    let mut in_channels = config.stage_channels[0];
    for stage_index in 0..config.stage_block_counts.len() {
        let stage_number = stage_index + 1;
        let out_channels = config.stage_channels[stage_index];
        let mut blocks = Vec::with_capacity(config.stage_block_counts[stage_index]);
        for block_index in 0..config.stage_block_counts[stage_index] {
            let stride = if block_index == 0 {
                config.stage_strides[stage_index]
            } else {
                1
            };
            let prefix = format!("layer{stage_number}.{block_index}");
            let conv1 = load_folded_conv(
                store,
                &format!("{prefix}.conv1"),
                &format!("{prefix}.bn1"),
                out_channels,
                in_channels,
                3,
                stride,
                1,
                config.batch_norm_epsilon,
            )?;
            let conv2 = load_folded_conv(
                store,
                &format!("{prefix}.conv2"),
                &format!("{prefix}.bn2"),
                out_channels,
                out_channels,
                3,
                1,
                1,
                config.batch_norm_epsilon,
            )?;
            let downsample = if stride != 1 || in_channels != out_channels {
                Some(load_folded_conv(
                    store,
                    &format!("{prefix}.downsample.0"),
                    &format!("{prefix}.downsample.1"),
                    out_channels,
                    in_channels,
                    1,
                    stride,
                    0,
                    config.batch_norm_epsilon,
                )?)
            } else {
                None
            };
            blocks.push(BasicBlockWeights {
                conv1,
                conv2,
                downsample,
            });
            in_channels = out_channels;
        }
        stages.push(blocks);
    }

    let fc_weight = store.required_f32(
        "fc.weight",
        &[config.num_classes, *config.stage_channels.last().unwrap()],
    )?;
    let fc_bias = store.required_f32("fc.bias", &[config.num_classes])?;

    Ok(ResNet18Weights {
        stem,
        stages,
        fc_weight,
        fc_bias,
    })
}

fn load_folded_conv(
    store: &TensorStore<'_>,
    conv_prefix: &str,
    bn_prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    eps: f32,
) -> Result<FoldedConv2d> {
    let conv_weight = store.required_f32(
        &format!("{conv_prefix}.weight"),
        &[out_channels, in_channels, kernel, kernel],
    )?;
    let conv_bias = store
        .optional_f32(&format!("{conv_prefix}.bias"), &[out_channels])?
        .unwrap_or_else(|| vec![0.0; out_channels]);
    let gamma = store.required_f32(&format!("{bn_prefix}.weight"), &[out_channels])?;
    let beta = store.required_f32(&format!("{bn_prefix}.bias"), &[out_channels])?;
    let running_mean = store.required_f32(&format!("{bn_prefix}.running_mean"), &[out_channels])?;
    let running_var = store.required_f32(&format!("{bn_prefix}.running_var"), &[out_channels])?;
    let (weight, bias) = fold_batch_norm_into_conv(
        &conv_weight,
        &conv_bias,
        &gamma,
        &beta,
        &running_mean,
        &running_var,
        eps,
        out_channels,
    );
    Ok(FoldedConv2d {
        weight,
        bias,
        shape: Conv2dShape {
            out_channels,
            in_channels,
            kernel_h: kernel,
            kernel_w: kernel,
        },
        stride,
        padding,
    })
}

pub fn fold_batch_norm_into_conv(
    conv_weight: &[f32],
    conv_bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
    out_channels: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(conv_bias.len(), out_channels);
    let values_per_out_channel = conv_weight.len() / out_channels;
    let mut folded_weight = conv_weight.to_vec();
    let mut folded_bias = vec![0.0; out_channels];
    for out_channel in 0..out_channels {
        let scale = gamma[out_channel] / (running_var[out_channel] + eps).sqrt();
        let start = out_channel * values_per_out_channel;
        let end = start + values_per_out_channel;
        for value in &mut folded_weight[start..end] {
            *value *= scale;
        }
        folded_bias[out_channel] =
            beta[out_channel] + (conv_bias[out_channel] - running_mean[out_channel]) * scale;
    }
    (folded_weight, folded_bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::cnn::{conv2d_nchw, Conv2dShape, NchwShape, NchwTensor};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual - *expected).abs() < 1e-5,
                "actual {actual} expected {expected}"
            );
        }
    }

    #[test]
    fn batch_norm_folding_matches_known_values() {
        let conv_weight = vec![1.0, 2.0, 3.0, 4.0];
        let conv_bias = vec![0.5, -1.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![0.25, -0.25];
        let running_mean = vec![0.5, 1.0];
        let running_var = vec![3.0, 0.25];
        let (weight, bias) = fold_batch_norm_into_conv(
            &conv_weight,
            &conv_bias,
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            1e-5,
            2,
        );

        let scale0 = 2.0 / (3.0f32 + 1e-5).sqrt();
        let scale1 = 0.5 / (0.25f32 + 1e-5).sqrt();
        assert_close(
            &weight,
            &[1.0 * scale0, 2.0 * scale0, 3.0 * scale1, 4.0 * scale1],
        );
        assert_close(
            &bias,
            &[0.25 + (0.5 - 0.5) * scale0, -0.25 + (-1.0 - 1.0) * scale1],
        );
    }

    #[test]
    fn batch_norm_folding_supports_no_original_bias() {
        let conv_bias = vec![0.0];
        let (_, bias) =
            fold_batch_norm_into_conv(&[2.0], &conv_bias, &[2.0], &[1.0], &[3.0], &[1.0], 0.0, 1);
        assert_close(&bias, &[-5.0]);
    }

    #[test]
    fn folded_conv_matches_unfused_conv_plus_batch_norm_output() {
        let input = NchwTensor::new(
            NchwShape {
                n: 1,
                c: 1,
                h: 2,
                w: 2,
            },
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let conv_weight = vec![2.0];
        let conv_bias = vec![0.5];
        let gamma = vec![1.5];
        let beta = vec![-0.25];
        let running_mean = vec![2.0];
        let running_var = vec![0.5];
        let eps = 1e-5f32;
        let shape = Conv2dShape {
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
        };

        let unfused_conv = conv2d_nchw(&input, &conv_weight, &conv_bias, shape, 1, 0);
        let scale = gamma[0] / (running_var[0] + eps).sqrt();
        let unfused = unfused_conv
            .data
            .iter()
            .map(|value| beta[0] + (*value - running_mean[0]) * scale)
            .collect::<Vec<_>>();

        let (folded_weight, folded_bias) = fold_batch_norm_into_conv(
            &conv_weight,
            &conv_bias,
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            eps,
            1,
        );
        let folded = conv2d_nchw(&input, &folded_weight, &folded_bias, shape, 1, 0);

        assert_close(&folded.data, &unfused);
    }
}
