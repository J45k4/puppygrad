use crate::models::vits::{
    conv1d, conv_transpose1d, Conv1dParams, ConvTranspose1dParams, VitsError,
};

use super::{BarkCodecConfig, BarkError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct BarkQuantizedLatents {
    pub frames: usize,
    pub codebook_dim: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecDecoderWeights {
    pub quantizer_codebooks: Vec<Vec<f32>>,
    pub initial: BarkEncodecConvWeights,
    pub lstm: BarkEncodecLstmWeights,
    pub upsample_blocks: Vec<BarkEncodecUpsampleBlockWeights>,
    pub final_conv: BarkEncodecConvWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecUpsampleBlockWeights {
    pub upsample: BarkEncodecConvTransposeWeights,
    pub residual_blocks: Vec<BarkEncodecResnetBlockWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecResnetBlockWeights {
    pub first: BarkEncodecConvWeights,
    pub second: BarkEncodecConvWeights,
    pub shortcut: Option<BarkEncodecConvWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecConvWeights {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecConvTransposeWeights {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecLstmWeights {
    pub layers: Vec<BarkEncodecLstmLayerWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarkEncodecLstmLayerWeights {
    pub weight_ih: Vec<f32>,
    pub weight_hh: Vec<f32>,
    pub bias_ih: Vec<f32>,
    pub bias_hh: Vec<f32>,
}

pub fn decode_encodec_audio(
    codes: &[Vec<usize>],
    config: &BarkCodecConfig,
    weights: &BarkEncodecDecoderWeights,
) -> Result<Vec<f32>> {
    let flat_codebooks = weights
        .quantizer_codebooks
        .iter()
        .flat_map(|codebook| codebook.iter().copied())
        .collect::<Vec<_>>();
    let latents = acoustic_codes_to_quantized_latents(
        codes,
        &flat_codebooks,
        config.codebook_size,
        config.codebook_dim,
    )?;
    decode_quantized_latents(&latents, config, weights)
}

pub fn decode_quantized_latents(
    latents: &BarkQuantizedLatents,
    config: &BarkCodecConfig,
    weights: &BarkEncodecDecoderWeights,
) -> Result<Vec<f32>> {
    if config.audio_channels != 1 {
        return Err(BarkError::Unsupported(format!(
            "native Bark EnCodec decode currently supports mono output, got {} channels",
            config.audio_channels
        )));
    }
    if latents.codebook_dim != config.hidden_size {
        return Err(BarkError::InvalidInput(format!(
            "quantized latent dim {} does not match codec hidden_size {}",
            latents.codebook_dim, config.hidden_size
        )));
    }

    let mut hidden = encodec_conv1d(
        &latents.values,
        latents.frames,
        &weights.initial,
        config.use_causal_conv,
        &config.pad_mode,
    )?;
    let mut frames = encodec_conv1d_output_len(latents.frames, &weights.initial)?;
    hidden = encodec_lstm(&hidden, frames, &weights.lstm)?;

    for block in &weights.upsample_blocks {
        elu_in_place(&mut hidden);
        hidden = encodec_conv_transpose1d(
            &hidden,
            frames,
            &block.upsample,
            config.use_causal_conv,
            config.trim_right_ratio,
        )?;
        frames = encodec_conv_transpose_output_len(
            frames,
            &block.upsample,
            config.use_causal_conv,
            config.trim_right_ratio,
        )?;
        for residual in &block.residual_blocks {
            hidden = encodec_resnet_block(&hidden, frames, residual, config)?;
        }
    }

    elu_in_place(&mut hidden);
    let audio = encodec_conv1d(
        &hidden,
        frames,
        &weights.final_conv,
        config.use_causal_conv,
        &config.pad_mode,
    )?;
    let audio_frames = encodec_conv1d_output_len(frames, &weights.final_conv)?;
    if audio.len() != config.audio_channels * audio_frames {
        return Err(BarkError::InvalidInput(format!(
            "decoded audio length {} does not match {} channels x {} frames",
            audio.len(),
            config.audio_channels,
            audio_frames
        )));
    }
    if audio.is_empty() || audio.iter().any(|sample| !sample.is_finite()) {
        return Err(BarkError::InvalidInput(
            "decoded audio must be finite and non-empty".to_string(),
        ));
    }
    // Hugging Face Bark returns the raw codec decoder output here. Keep samples
    // unchanged; PCM writing is the only later conversion boundary.
    Ok(audio)
}

pub fn acoustic_codes_to_quantized_latents(
    codes: &[Vec<usize>],
    codebook_embeddings: &[f32],
    codebook_size: usize,
    codebook_dim: usize,
) -> Result<BarkQuantizedLatents> {
    if codes.is_empty() {
        return Err(BarkError::InvalidInput(
            "acoustic codes must contain at least one codebook".to_string(),
        ));
    }
    if codebook_size == 0 || codebook_dim == 0 {
        return Err(BarkError::InvalidInput(
            "codebook_size and codebook_dim must be > 0".to_string(),
        ));
    }
    let expected_embedding_len = codes.len() * codebook_size * codebook_dim;
    if codebook_embeddings.len() != expected_embedding_len {
        return Err(BarkError::InvalidInput(format!(
            "codebook embedding length {} does not match {} codebooks x {codebook_size} entries x {codebook_dim} dims",
            codebook_embeddings.len(),
            codes.len()
        )));
    }
    let frames = codes[0].len();
    if frames == 0 {
        return Err(BarkError::InvalidInput(
            "acoustic codes must contain at least one frame".to_string(),
        ));
    }
    if codes.iter().any(|row| row.len() != frames) {
        return Err(BarkError::InvalidInput(
            "all acoustic codebooks must have equal frame counts".to_string(),
        ));
    }

    let mut values = vec![0.0; frames * codebook_dim];
    for (codebook, row) in codes.iter().enumerate() {
        for (frame, &code) in row.iter().enumerate() {
            if code >= codebook_size {
                return Err(BarkError::InvalidInput(format!(
                    "acoustic code {code} exceeds codebook size {codebook_size}"
                )));
            }
            let embedding_start = (codebook * codebook_size + code) * codebook_dim;
            for dim in 0..codebook_dim {
                values[dim * frames + frame] += codebook_embeddings[embedding_start + dim];
            }
        }
    }
    Ok(BarkQuantizedLatents {
        frames,
        codebook_dim,
        values,
    })
}

fn encodec_resnet_block(
    input: &[f32],
    frames: usize,
    weights: &BarkEncodecResnetBlockWeights,
    config: &BarkCodecConfig,
) -> Result<Vec<f32>> {
    let mut hidden = input.to_vec();
    elu_in_place(&mut hidden);
    hidden = encodec_conv1d(
        &hidden,
        frames,
        &weights.first,
        config.use_causal_conv,
        &config.pad_mode,
    )?;
    let hidden_frames = encodec_conv1d_output_len(frames, &weights.first)?;
    elu_in_place(&mut hidden);
    hidden = encodec_conv1d(
        &hidden,
        hidden_frames,
        &weights.second,
        config.use_causal_conv,
        &config.pad_mode,
    )?;
    let output_frames = encodec_conv1d_output_len(hidden_frames, &weights.second)?;
    let residual = if let Some(shortcut) = &weights.shortcut {
        encodec_conv1d(
            input,
            frames,
            shortcut,
            config.use_causal_conv,
            &config.pad_mode,
        )?
    } else {
        input.to_vec()
    };
    if residual.len() != hidden.len() {
        return Err(BarkError::InvalidInput(format!(
            "Encodec residual length mismatch: shortcut {} vs block {} at {output_frames} frames",
            residual.len(),
            hidden.len()
        )));
    }
    for (dst, src) in hidden.iter_mut().zip(residual) {
        *dst += src;
    }
    Ok(hidden)
}

fn encodec_conv1d(
    input: &[f32],
    input_len: usize,
    weights: &BarkEncodecConvWeights,
    causal: bool,
    pad_mode: &str,
) -> Result<Vec<f32>> {
    let effective_kernel = (weights.kernel_size - 1) * weights.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(weights.stride);
    let extra_padding = extra_padding_for_conv1d(input_len, effective_kernel, weights.stride);
    let (left, right) = if causal {
        (padding_total, extra_padding)
    } else {
        let right = padding_total / 2;
        (padding_total - right, right + extra_padding)
    };
    let padded = pad_channel_major(input, weights.in_channels, input_len, left, right, pad_mode)?;
    conv1d(
        &padded,
        input_len + left + right,
        Conv1dParams {
            in_channels: weights.in_channels,
            out_channels: weights.out_channels,
            kernel_size: weights.kernel_size,
            stride: weights.stride,
            padding: 0,
            dilation: weights.dilation,
            groups: 1,
        },
        &weights.weight,
        Some(&weights.bias),
    )
    .map_err(vits_error)
}

fn encodec_conv1d_output_len(input_len: usize, weights: &BarkEncodecConvWeights) -> Result<usize> {
    let effective_kernel = (weights.kernel_size - 1) * weights.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(weights.stride);
    let extra_padding = extra_padding_for_conv1d(input_len, effective_kernel, weights.stride);
    let padded_len = input_len + padding_total + extra_padding;
    Conv1dParams {
        in_channels: weights.in_channels,
        out_channels: weights.out_channels,
        kernel_size: weights.kernel_size,
        stride: weights.stride,
        padding: 0,
        dilation: weights.dilation,
        groups: 1,
    }
    .output_len(padded_len)
    .map_err(vits_error)
}

fn encodec_conv_transpose1d(
    input: &[f32],
    input_len: usize,
    weights: &BarkEncodecConvTransposeWeights,
    causal: bool,
    trim_right_ratio: f32,
) -> Result<Vec<f32>> {
    let raw = conv_transpose1d(
        input,
        input_len,
        ConvTranspose1dParams {
            in_channels: weights.in_channels,
            out_channels: weights.out_channels,
            kernel_size: weights.kernel_size,
            stride: weights.stride,
            padding: 0,
            dilation: 1,
            groups: 1,
            output_padding: 0,
        },
        &weights.weight,
        Some(&weights.bias),
    )
    .map_err(vits_error)?;
    let raw_frames = ConvTranspose1dParams {
        in_channels: weights.in_channels,
        out_channels: weights.out_channels,
        kernel_size: weights.kernel_size,
        stride: weights.stride,
        padding: 0,
        dilation: 1,
        groups: 1,
        output_padding: 0,
    }
    .output_len(input_len)
    .map_err(vits_error)?;
    let output_frames =
        encodec_conv_transpose_output_len(input_len, weights, causal, trim_right_ratio)?;
    let padding_total = weights.kernel_size - weights.stride;
    let (start, end) = if causal {
        let trim_right = (padding_total as f32 * trim_right_ratio).ceil() as usize;
        (0, raw_frames.saturating_sub(trim_right))
    } else {
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        (padding_left, raw_frames.saturating_sub(padding_right))
    };
    if end < start || end - start != output_frames {
        return Err(BarkError::InvalidInput(format!(
            "Encodec conv_transpose trim produced invalid frame range {start}..{end} from {raw_frames}"
        )));
    }
    let mut trimmed = vec![0.0; weights.out_channels * output_frames];
    for channel in 0..weights.out_channels {
        let raw_start = channel * raw_frames + start;
        let dst_start = channel * output_frames;
        trimmed[dst_start..dst_start + output_frames]
            .copy_from_slice(&raw[raw_start..raw_start + output_frames]);
    }
    Ok(trimmed)
}

fn encodec_conv_transpose_output_len(
    input_len: usize,
    weights: &BarkEncodecConvTransposeWeights,
    causal: bool,
    trim_right_ratio: f32,
) -> Result<usize> {
    let raw = ConvTranspose1dParams {
        in_channels: weights.in_channels,
        out_channels: weights.out_channels,
        kernel_size: weights.kernel_size,
        stride: weights.stride,
        padding: 0,
        dilation: 1,
        groups: 1,
        output_padding: 0,
    }
    .output_len(input_len)
    .map_err(vits_error)?;
    let padding_total = weights.kernel_size - weights.stride;
    let trim = if causal {
        (padding_total as f32 * trim_right_ratio).ceil() as usize
    } else {
        padding_total
    };
    Ok(raw.saturating_sub(trim))
}

fn encodec_lstm(
    input: &[f32],
    frames: usize,
    weights: &BarkEncodecLstmWeights,
) -> Result<Vec<f32>> {
    if weights.layers.is_empty() {
        return Ok(input.to_vec());
    }
    let channels = infer_lstm_channels(&weights.layers[0])?;
    if input.len() != channels * frames {
        return Err(BarkError::InvalidInput(format!(
            "LSTM input length {} does not match {} channels x {} frames",
            input.len(),
            channels,
            frames
        )));
    }
    let residual = input.to_vec();
    let mut layer_input = input.to_vec();
    for layer in &weights.layers {
        let layer_channels = infer_lstm_channels(layer)?;
        if layer_channels != channels {
            return Err(BarkError::InvalidInput(
                "all EnCodec LSTM layers must use the same hidden size".to_string(),
            ));
        }
        layer_input = encodec_lstm_layer(&layer_input, frames, layer, channels)?;
    }
    let mut output = layer_input;
    for (dst, src) in output.iter_mut().zip(residual) {
        *dst += src;
    }
    Ok(output)
}

fn encodec_lstm_layer(
    input: &[f32],
    frames: usize,
    weights: &BarkEncodecLstmLayerWeights,
    channels: usize,
) -> Result<Vec<f32>> {
    validate_lstm_layer(weights, channels)?;
    let mut output = vec![0.0; channels * frames];
    let mut h = vec![0.0f32; channels];
    let mut c = vec![0.0f32; channels];
    for frame in 0..frames {
        let x = (0..channels)
            .map(|channel| input[channel * frames + frame])
            .collect::<Vec<_>>();
        let mut gates = vec![0.0f32; channels * 4];
        for gate in 0..(channels * 4) {
            let mut sum = weights.bias_ih[gate] + weights.bias_hh[gate];
            for input_channel in 0..channels {
                sum += weights.weight_ih[gate * channels + input_channel] * x[input_channel];
                sum += weights.weight_hh[gate * channels + input_channel] * h[input_channel];
            }
            gates[gate] = sum;
        }
        for channel in 0..channels {
            let i = sigmoid(gates[channel]);
            let f = sigmoid(gates[channels + channel]);
            let g = gates[2 * channels + channel].tanh();
            let o = sigmoid(gates[3 * channels + channel]);
            c[channel] = f * c[channel] + i * g;
            h[channel] = o * c[channel].tanh();
            output[channel * frames + frame] = h[channel];
        }
    }
    Ok(output)
}

fn validate_lstm_layer(weights: &BarkEncodecLstmLayerWeights, channels: usize) -> Result<()> {
    let gate = channels * 4;
    let matrix = gate * channels;
    if weights.weight_ih.len() != matrix
        || weights.weight_hh.len() != matrix
        || weights.bias_ih.len() != gate
        || weights.bias_hh.len() != gate
    {
        return Err(BarkError::InvalidInput(
            "invalid EnCodec LSTM tensor lengths".to_string(),
        ));
    }
    Ok(())
}

fn infer_lstm_channels(weights: &BarkEncodecLstmLayerWeights) -> Result<usize> {
    if weights.bias_ih.is_empty() || !weights.bias_ih.len().is_multiple_of(4) {
        return Err(BarkError::InvalidInput(
            "LSTM bias length must be divisible by 4".to_string(),
        ));
    }
    Ok(weights.bias_ih.len() / 4)
}

fn pad_channel_major(
    input: &[f32],
    channels: usize,
    frames: usize,
    left: usize,
    right: usize,
    mode: &str,
) -> Result<Vec<f32>> {
    if input.len() != channels * frames {
        return Err(BarkError::InvalidInput(format!(
            "padding input length {} does not match {} channels x {} frames",
            input.len(),
            channels,
            frames
        )));
    }
    let padded_frames = frames + left + right;
    let reflect_extra_pad = if mode == "reflect" {
        let max_pad = left.max(right);
        if frames <= max_pad {
            max_pad - frames + 1
        } else {
            0
        }
    } else {
        0
    };
    let reflect_frames = frames + reflect_extra_pad;
    let mut output = vec![0.0; channels * padded_frames];
    for channel in 0..channels {
        for dst_t in 0..padded_frames {
            let value = match mode {
                "constant" => {
                    if dst_t < left || dst_t >= left + frames {
                        0.0
                    } else {
                        input[channel * frames + (dst_t - left)]
                    }
                }
                "reflect" => {
                    let src_t = reflect_index(dst_t as isize - left as isize, reflect_frames);
                    if src_t < frames {
                        input[channel * frames + src_t]
                    } else {
                        0.0
                    }
                }
                other => {
                    return Err(BarkError::Unsupported(format!(
                        "unsupported EnCodec pad mode {other:?}"
                    )))
                }
            };
            output[channel * padded_frames + dst_t] = value;
        }
    }
    Ok(output)
}

fn reflect_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * len - 2) as isize;
    let mut value = index % period;
    if value < 0 {
        value += period;
    }
    if value >= len as isize {
        (period - value) as usize
    } else {
        value as usize
    }
}

fn extra_padding_for_conv1d(input_len: usize, kernel_size: usize, stride: usize) -> usize {
    let padding_total = kernel_size.saturating_sub(stride);
    let n_frames = ((input_len + padding_total).saturating_sub(kernel_size) + stride) / stride;
    let ideal_length = n_frames.saturating_sub(1) * stride + kernel_size - padding_total;
    ideal_length.saturating_sub(input_len)
}

fn elu_in_place(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = value.exp() - 1.0;
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn vits_error(error: VitsError) -> BarkError {
    BarkError::InvalidInput(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_codebook_embeddings_per_frame() -> Result<()> {
        let codes = vec![vec![0, 1, 0], vec![1, 0, 1]];
        let embeddings = vec![
            1.0, 0.0, // codebook 0 code 0
            0.0, 1.0, // codebook 0 code 1
            2.0, 0.0, // codebook 1 code 0
            0.0, 2.0, // codebook 1 code 1
        ];

        let latents = acoustic_codes_to_quantized_latents(&codes, &embeddings, 2, 2)?;

        assert_eq!(latents.frames, 3);
        assert_eq!(latents.codebook_dim, 2);
        assert_eq!(latents.values, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn decode_quantized_latents_runs_tiny_decoder() -> Result<()> {
        let config = tiny_codec_config();
        let weights = tiny_decoder_weights(1.0);
        let codes = vec![vec![0, 1]];

        let audio = decode_encodec_audio(&codes, &config, &weights)?;

        assert!(!audio.is_empty());
        assert!(audio.iter().all(|sample| sample.is_finite()));
        Ok(())
    }

    #[test]
    fn decode_quantized_latents_preserves_raw_decoder_amplitude_without_clamp() -> Result<()> {
        let config = tiny_codec_config();
        let weights = tiny_decoder_weights(8.0);
        let codes = vec![vec![0, 1]];

        let audio = decode_encodec_audio(&codes, &config, &weights)?;

        assert!(audio.iter().any(|sample| sample.abs() > 1.0), "{audio:?}");
        assert!(audio.iter().all(|sample| sample.is_finite()));
        Ok(())
    }

    #[test]
    fn reflect_padding_matches_encodec_small_input_zero_extension() -> Result<()> {
        let padded = pad_channel_major(&[3.0], 1, 1, 2, 0, "reflect")?;

        assert_eq!(padded, vec![0.0, 0.0, 3.0]);
        Ok(())
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

    fn tiny_decoder_weights(final_scale: f32) -> BarkEncodecDecoderWeights {
        BarkEncodecDecoderWeights {
            quantizer_codebooks: vec![vec![0.5, -0.25, 0.25, 0.5]],
            initial: conv(2, 2, 1, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]),
            lstm: BarkEncodecLstmWeights {
                layers: vec![zero_lstm(2)],
            },
            upsample_blocks: vec![BarkEncodecUpsampleBlockWeights {
                upsample: BarkEncodecConvTransposeWeights {
                    in_channels: 2,
                    out_channels: 1,
                    kernel_size: 4,
                    stride: 2,
                    weight: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    bias: vec![0.0],
                },
                residual_blocks: vec![BarkEncodecResnetBlockWeights {
                    first: conv(1, 1, 1, vec![0.0], vec![0.0]),
                    second: conv(1, 1, 1, vec![0.0], vec![0.0]),
                    shortcut: Some(conv(1, 1, 1, vec![1.0], vec![0.0])),
                }],
            }],
            final_conv: conv(1, 1, 1, vec![final_scale], vec![0.0]),
        }
    }

    fn conv(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> BarkEncodecConvWeights {
        BarkEncodecConvWeights {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            dilation: 1,
            weight,
            bias,
        }
    }

    fn zero_lstm(channels: usize) -> BarkEncodecLstmLayerWeights {
        BarkEncodecLstmLayerWeights {
            weight_ih: vec![0.0; channels * 4 * channels],
            weight_hh: vec![0.0; channels * 4 * channels],
            bias_ih: vec![0.0; channels * 4],
            bias_hh: vec![0.0; channels * 4],
        }
    }
}
