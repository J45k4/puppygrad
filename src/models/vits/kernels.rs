use super::{Result, VitsError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dParams {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1dParams {
    pub fn output_len(self, input_len: usize) -> Result<usize> {
        self.validate()?;
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        if input_len + 2 * self.padding < effective_kernel {
            return Ok(0);
        }
        Ok((input_len + 2 * self.padding - effective_kernel) / self.stride + 1)
    }

    pub fn weight_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.out_channels * (self.in_channels / self.groups) * self.kernel_size)
    }

    fn validate(self) -> Result<()> {
        if self.in_channels == 0 || self.out_channels == 0 {
            return Err(VitsError::InvalidInput(
                "conv1d channel counts must be > 0".to_string(),
            ));
        }
        if self.kernel_size == 0 || self.stride == 0 || self.dilation == 0 {
            return Err(VitsError::InvalidInput(
                "conv1d kernel_size, stride, and dilation must be > 0".to_string(),
            ));
        }
        if self.groups == 0 {
            return Err(VitsError::InvalidInput(
                "conv1d groups must be > 0".to_string(),
            ));
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            return Err(VitsError::InvalidInput(format!(
                "in_channels {} must be divisible by groups {}",
                self.in_channels, self.groups
            )));
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            return Err(VitsError::InvalidInput(format!(
                "out_channels {} must be divisible by groups {}",
                self.out_channels, self.groups
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose1dParams {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub output_padding: usize,
}

impl ConvTranspose1dParams {
    pub fn output_len(self, input_len: usize) -> Result<usize> {
        self.validate()?;
        if input_len == 0 {
            return Ok(0);
        }
        Ok((input_len - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + self.output_padding
            + 1)
    }

    pub fn weight_len(self) -> Result<usize> {
        self.validate()?;
        Ok(self.in_channels * (self.out_channels / self.groups) * self.kernel_size)
    }

    fn validate(self) -> Result<()> {
        if self.in_channels == 0 || self.out_channels == 0 {
            return Err(VitsError::InvalidInput(
                "conv_transpose1d channel counts must be > 0".to_string(),
            ));
        }
        if self.kernel_size == 0 || self.stride == 0 || self.dilation == 0 {
            return Err(VitsError::InvalidInput(
                "conv_transpose1d kernel_size, stride, and dilation must be > 0".to_string(),
            ));
        }
        if self.groups == 0 {
            return Err(VitsError::InvalidInput(
                "conv_transpose1d groups must be > 0".to_string(),
            ));
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            return Err(VitsError::InvalidInput(format!(
                "in_channels {} must be divisible by groups {}",
                self.in_channels, self.groups
            )));
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            return Err(VitsError::InvalidInput(format!(
                "out_channels {} must be divisible by groups {}",
                self.out_channels, self.groups
            )));
        }
        if self.output_padding >= self.stride {
            return Err(VitsError::InvalidInput(format!(
                "output_padding {} must be less than stride {}",
                self.output_padding, self.stride
            )));
        }
        Ok(())
    }
}

pub fn same_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

pub fn conv1d(
    input: &[f32],
    input_len: usize,
    params: Conv1dParams,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let output_len = params.output_len(input_len)?;
    if input.len() != params.in_channels * input_len {
        return Err(VitsError::InvalidInput(format!(
            "conv1d input length {} does not match {} channels x {} frames",
            input.len(),
            params.in_channels,
            input_len
        )));
    }
    if weight.len() != params.weight_len()? {
        return Err(VitsError::InvalidInput(format!(
            "conv1d weight length {} does not match expected {}",
            weight.len(),
            params.weight_len()?
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != params.out_channels {
            return Err(VitsError::InvalidInput(format!(
                "conv1d bias length {} does not match out_channels {}",
                bias.len(),
                params.out_channels
            )));
        }
    }

    let mut out = vec![0.0f32; params.out_channels * output_len];
    let in_per_group = params.in_channels / params.groups;
    let out_per_group = params.out_channels / params.groups;
    for group in 0..params.groups {
        for ocg in 0..out_per_group {
            let oc = group * out_per_group + ocg;
            for ot in 0..output_len {
                let mut sum = bias.map_or(0.0, |b| b[oc]);
                for icg in 0..in_per_group {
                    let ic = group * in_per_group + icg;
                    for k in 0..params.kernel_size {
                        let padded = ot * params.stride + k * params.dilation;
                        if padded < params.padding {
                            continue;
                        }
                        let it = padded - params.padding;
                        if it >= input_len {
                            continue;
                        }
                        let x = input[ic * input_len + it];
                        let w = weight[(oc * in_per_group + icg) * params.kernel_size + k];
                        sum += x * w;
                    }
                }
                out[oc * output_len + ot] = sum;
            }
        }
    }
    Ok(out)
}

pub fn conv_transpose1d(
    input: &[f32],
    input_len: usize,
    params: ConvTranspose1dParams,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let output_len = params.output_len(input_len)?;
    if input.len() != params.in_channels * input_len {
        return Err(VitsError::InvalidInput(format!(
            "conv_transpose1d input length {} does not match {} channels x {} frames",
            input.len(),
            params.in_channels,
            input_len
        )));
    }
    if weight.len() != params.weight_len()? {
        return Err(VitsError::InvalidInput(format!(
            "conv_transpose1d weight length {} does not match expected {}",
            weight.len(),
            params.weight_len()?
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != params.out_channels {
            return Err(VitsError::InvalidInput(format!(
                "conv_transpose1d bias length {} does not match out_channels {}",
                bias.len(),
                params.out_channels
            )));
        }
    }

    let mut out = vec![0.0f32; params.out_channels * output_len];
    let in_per_group = params.in_channels / params.groups;
    let out_per_group = params.out_channels / params.groups;
    for group in 0..params.groups {
        for icg in 0..in_per_group {
            let ic = group * in_per_group + icg;
            for it in 0..input_len {
                let x = input[ic * input_len + it];
                for ocg in 0..out_per_group {
                    let oc = group * out_per_group + ocg;
                    for k in 0..params.kernel_size {
                        let raw = it * params.stride + k * params.dilation;
                        if raw < params.padding {
                            continue;
                        }
                        let ot = raw - params.padding;
                        if ot >= output_len {
                            continue;
                        }
                        let w = weight[(ic * out_per_group + ocg) * params.kernel_size + k];
                        out[oc * output_len + ot] += x * w;
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        for oc in 0..params.out_channels {
            let row = &mut out[oc * output_len..(oc + 1) * output_len];
            for value in row {
                *value += bias[oc];
            }
        }
    }

    Ok(out)
}

pub fn leaky_relu_in_place(values: &mut [f32], negative_slope: f32) {
    for value in values {
        if *value < 0.0 {
            *value *= negative_slope;
        }
    }
}

pub fn channel_layer_norm_in_place(
    values: &mut [f32],
    channels: usize,
    frames: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    if values.len() != channels * frames {
        return Err(VitsError::InvalidInput(format!(
            "layer norm values length {} does not match {} channels x {} frames",
            values.len(),
            channels,
            frames
        )));
    }
    if gamma.len() != channels || beta.len() != channels {
        return Err(VitsError::InvalidInput(format!(
            "layer norm gamma/beta lengths {}/{} do not match channels {}",
            gamma.len(),
            beta.len(),
            channels
        )));
    }
    if eps <= 0.0 {
        return Err(VitsError::InvalidInput(
            "layer norm eps must be > 0".to_string(),
        ));
    }

    for t in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..channels {
            mean += values[c * frames + t];
        }
        mean /= channels as f32;

        let mut variance = 0.0f32;
        for c in 0..channels {
            let delta = values[c * frames + t] - mean;
            variance += delta * delta;
        }
        variance /= channels as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        for c in 0..channels {
            let index = c * frames + t;
            values[index] = (values[index] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
    Ok(())
}

pub fn gated_tanh_sigmoid(values: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
    if values.len() != channels * 2 * frames {
        return Err(VitsError::InvalidInput(format!(
            "gated activation input length {} does not match 2 x {} channels x {} frames",
            values.len(),
            channels,
            frames
        )));
    }

    let mut out = vec![0.0f32; channels * frames];
    for c in 0..channels {
        for t in 0..frames {
            let a = values[c * frames + t].tanh();
            let b = sigmoid(values[(channels + c) * frames + t]);
            out[c * frames + t] = a * b;
        }
    }
    Ok(out)
}

pub fn elementwise_affine(
    values: &[f32],
    channels: usize,
    frames: usize,
    scale: &[f32],
    bias: &[f32],
    reverse: bool,
) -> Result<Vec<f32>> {
    if values.len() != channels * frames {
        return Err(VitsError::InvalidInput(format!(
            "affine values length {} does not match {} channels x {} frames",
            values.len(),
            channels,
            frames
        )));
    }
    if scale.len() != channels || bias.len() != channels {
        return Err(VitsError::InvalidInput(format!(
            "affine scale/bias lengths {}/{} do not match channels {}",
            scale.len(),
            bias.len(),
            channels
        )));
    }

    let mut out = vec![0.0f32; values.len()];
    for c in 0..channels {
        if reverse && scale[c] == 0.0 {
            return Err(VitsError::InvalidInput(format!(
                "affine scale for channel {c} is zero"
            )));
        }
        for t in 0..frames {
            let index = c * frames + t;
            out[index] = if reverse {
                (values[index] - bias[c]) / scale[c]
            } else {
                values[index] * scale[c] + bias[c]
            };
        }
    }
    Ok(out)
}

pub fn flip_channels(values: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
    if values.len() != channels * frames {
        return Err(VitsError::InvalidInput(format!(
            "flip values length {} does not match {} channels x {} frames",
            values.len(),
            channels,
            frames
        )));
    }
    let mut out = vec![0.0f32; values.len()];
    for c in 0..channels {
        let src = c * frames;
        let dst = (channels - 1 - c) * frames;
        out[dst..dst + frames].copy_from_slice(&values[src..src + frames]);
    }
    Ok(out)
}

pub fn residual_coupling_reverse(
    values: &[f32],
    channels: usize,
    frames: usize,
    shift: &[f32],
    log_scale: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if !channels.is_multiple_of(2) {
        return Err(VitsError::InvalidInput(format!(
            "residual coupling channels {channels} must be even"
        )));
    }
    if values.len() != channels * frames {
        return Err(VitsError::InvalidInput(format!(
            "residual coupling values length {} does not match {} channels x {} frames",
            values.len(),
            channels,
            frames
        )));
    }
    let half = channels / 2;
    if shift.len() != half * frames {
        return Err(VitsError::InvalidInput(format!(
            "residual coupling shift length {} does not match {} channels x {} frames",
            shift.len(),
            half,
            frames
        )));
    }
    if let Some(log_scale) = log_scale {
        if log_scale.len() != half * frames {
            return Err(VitsError::InvalidInput(format!(
                "residual coupling log_scale length {} does not match {} channels x {} frames",
                log_scale.len(),
                half,
                frames
            )));
        }
    }

    let mut out = values.to_vec();
    for c in 0..half {
        for t in 0..frames {
            let src = (half + c) * frames + t;
            let flow = c * frames + t;
            let scale = log_scale.map_or(1.0, |logs| (-logs[flow]).exp());
            out[src] = (values[src] - shift[flow]) * scale;
        }
    }
    Ok(out)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_matches_small_reference() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ];
        let weight = vec![
            1.0, 0.0, -1.0, //
            0.5, 0.5, 0.5,
        ];
        let out = conv1d(
            &input,
            4,
            Conv1dParams {
                in_channels: 2,
                out_channels: 1,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            &weight,
            Some(&[0.25]),
        )
        .unwrap();

        assert_eq!(out, vec![3.75, 7.25, 8.75, 10.75]);
    }

    #[test]
    fn conv1d_supports_grouped_depthwise_case() {
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let weight = vec![1.0, 1.0, 0.1, 0.1];
        let out = conv1d(
            &input,
            3,
            Conv1dParams {
                in_channels: 2,
                out_channels: 2,
                kernel_size: 2,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 2,
            },
            &weight,
            None,
        )
        .unwrap();

        assert_eq!(out, vec![3.0, 5.0, 3.0, 5.0]);
    }

    #[test]
    fn conv_transpose1d_matches_small_reference() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.5, -1.0];
        let out = conv_transpose1d(
            &input,
            3,
            ConvTranspose1dParams {
                in_channels: 1,
                out_channels: 1,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            &weight,
            Some(&[0.25]),
        )
        .unwrap();

        assert_eq!(out, vec![0.75, 1.25, 1.25, 1.25, 1.75]);
    }

    #[test]
    fn same_padding_matches_vits_formula() {
        assert_eq!(same_padding(3, 1), 1);
        assert_eq!(same_padding(5, 2), 4);
    }

    #[test]
    fn channel_layer_norm_normalizes_each_frame_across_channels() {
        let mut values = vec![1.0, 3.0, 5.0, 7.0];

        channel_layer_norm_in_place(&mut values, 2, 2, &[1.0, 1.0], &[0.0, 0.0], 1e-5).unwrap();

        assert!((values[0] + 0.999_995).abs() < 1e-4);
        assert!((values[1] + 0.999_995).abs() < 1e-4);
        assert!((values[2] - 0.999_995).abs() < 1e-4);
        assert!((values[3] - 0.999_995).abs() < 1e-4);
    }

    #[test]
    fn gated_tanh_sigmoid_applies_vits_activation() {
        let values = vec![0.0, 1.0, 0.0, 0.0];
        let out = gated_tanh_sigmoid(&values, 1, 2).unwrap();

        assert_eq!(out[0], 0.0);
        assert!((out[1] - 1.0f32.tanh() * 0.5).abs() < 1e-6);
    }

    #[test]
    fn elementwise_affine_reverses_transform() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let encoded = elementwise_affine(&values, 2, 2, &[2.0, 4.0], &[1.0, -1.0], false).unwrap();
        let decoded = elementwise_affine(&encoded, 2, 2, &[2.0, 4.0], &[1.0, -1.0], true).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn flip_channels_reverses_channel_order() {
        let values = vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0];

        let out = flip_channels(&values, 3, 2).unwrap();

        assert_eq!(out, vec![100.0, 200.0, 10.0, 20.0, 1.0, 2.0]);
    }

    #[test]
    fn residual_coupling_reverse_updates_second_half() {
        let values = vec![1.0, 2.0, 10.0, 20.0];
        let out = residual_coupling_reverse(&values, 2, 2, &[3.0, 4.0], None).unwrap();

        assert_eq!(out, vec![1.0, 2.0, 7.0, 16.0]);
    }
}
