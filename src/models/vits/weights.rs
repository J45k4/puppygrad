use std::path::Path;

use crate::models::onnx::{load_onnx_initializers, OnnxInitializerStore, OnnxTensorType};

use super::{
    channel_layer_norm_in_place, conv1d, conv_transpose1d, expand_by_durations, flip_channels,
    gated_tanh_sigmoid, leaky_relu_in_place, log_durations_to_durations, residual_coupling_reverse,
    same_padding, Conv1dParams, ConvTranspose1dParams, DeterministicDurationPredictorWeights,
    DeterministicRng, Result, VitsError, VitsSynthesisScales,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VitsWeightConfig {
    pub num_symbols: usize,
    pub num_speakers: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvWeights {
    pub weight: DenseTensor,
    pub bias: Option<DenseTensor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormWeights {
    pub gamma: DenseTensor,
    pub beta: DenseTensor,
}

impl LayerNormWeights {
    fn apply(&self, values: &mut [f32], channels: usize, frames: usize) -> Result<()> {
        channel_layer_norm_in_place(
            values,
            channels,
            frames,
            &self.gamma.values,
            &self.beta.values,
            1e-5,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VitsWeights {
    pub hidden_channels: usize,
    pub text_encoder: TextEncoderWeights,
    pub deterministic_duration_predictor: Option<DeterministicDurationPredictorWeights>,
    pub duration_predictor: StochasticDurationPredictorWeights,
    pub residual_coupling_flow: ResidualCouplingFlowWeights,
    pub generator: GeneratorWeights,
    pub speaker_embedding: Option<DenseTensor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VitsInferenceOutput {
    pub samples: Vec<f32>,
    pub durations: Vec<usize>,
    pub acoustic_frames: usize,
}

impl VitsWeights {
    pub fn infer_phoneme_ids(
        &self,
        phoneme_ids: &[usize],
        scales: VitsSynthesisScales,
        speaker_id: Option<usize>,
        rng: &mut DeterministicRng,
    ) -> Result<VitsInferenceOutput> {
        if let Some(speaker_id) = speaker_id {
            let Some(embedding) = &self.speaker_embedding else {
                if speaker_id == 0 {
                    // Single-speaker exports commonly pass speaker 0 even without an embedding.
                } else {
                    return Err(VitsError::Unsupported(format!(
                        "speaker id {speaker_id} was requested, but this voice has no speaker embedding"
                    )));
                }
                return self.infer_phoneme_ids(phoneme_ids, scales, None, rng);
            };
            if speaker_id >= embedding.shape[0] {
                return Err(VitsError::InvalidInput(format!(
                    "speaker id {speaker_id} is out of range for {} speaker(s)",
                    embedding.shape[0]
                )));
            }
            return Err(VitsError::Unsupported(
                "multi-speaker conditioning is loaded but not wired into native inference yet"
                    .to_string(),
            ));
        }

        let encoded = self.text_encoder.infer(phoneme_ids)?;
        let log_durations = self.duration_predictor.reverse_log_durations(
            &encoded.hidden,
            encoded.frames,
            scales.noise_w,
            rng,
        )?;
        let durations = log_durations_to_durations(&log_durations, scales.length_scale);
        let prior = encoded.expand_prior_by_durations(&durations)?;
        let mut latent = vec![0.0f32; prior.channels * prior.frames];
        for (index, value) in latent.iter_mut().enumerate() {
            *value = prior.mean[index]
                + rng.standard_normal() * scales.noise_scale * prior.log_scale[index].exp();
        }
        let latent = self
            .residual_coupling_flow
            .reverse(&latent, prior.channels, prior.frames)?;
        let samples = self.generator.infer(&latent, prior.frames)?;
        Ok(VitsInferenceOutput {
            samples,
            durations,
            acoustic_frames: prior.frames,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderWeights {
    pub embedding: DenseTensor,
    pub projection: ConvWeights,
    pub attention_layers: Vec<TextEncoderAttentionLayerWeights>,
    pub ffn_layers: Vec<TextEncoderFfnLayerWeights>,
    pub norm_layers_1: Vec<LayerNormWeights>,
    pub norm_layers_2: Vec<LayerNormWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderOutput {
    pub hidden: Vec<f32>,
    pub prior_mean: Vec<f32>,
    pub prior_log_scale: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedPrior {
    pub mean: Vec<f32>,
    pub log_scale: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
}

impl TextEncoderOutput {
    pub fn expand_prior_by_durations(&self, durations: &[usize]) -> Result<ExpandedPrior> {
        if durations.len() != self.frames {
            return Err(VitsError::InvalidInput(format!(
                "duration count {} does not match text frames {}",
                durations.len(),
                self.frames
            )));
        }
        let frames = durations.iter().sum::<usize>();
        if frames == 0 {
            return Err(VitsError::InvalidInput(
                "expanded prior must contain at least one frame".to_string(),
            ));
        }
        Ok(ExpandedPrior {
            mean: expand_channel_major_by_durations(&self.prior_mean, self.channels, durations)?,
            log_scale: expand_channel_major_by_durations(
                &self.prior_log_scale,
                self.channels,
                durations,
            )?,
            channels: self.channels,
            frames,
        })
    }
}

impl TextEncoderWeights {
    pub fn infer(&self, phoneme_ids: &[usize]) -> Result<TextEncoderOutput> {
        if phoneme_ids.is_empty() {
            return Err(VitsError::InvalidInput(
                "text encoder input must not be empty".to_string(),
            ));
        }
        let hidden_channels = embedding_hidden_channels(&self.embedding)?;
        let frames = phoneme_ids.len();
        if self.ffn_layers.len() != self.attention_layers.len()
            || self.norm_layers_1.len() != self.attention_layers.len()
            || self.norm_layers_2.len() != self.attention_layers.len()
        {
            return Err(VitsError::InvalidInput(format!(
                "text encoder has mismatched layer counts: attention={}, ffn={}, norm1={}, norm2={}",
                self.attention_layers.len(),
                self.ffn_layers.len(),
                self.norm_layers_1.len(),
                self.norm_layers_2.len()
            )));
        }
        let mut hidden = vec![0.0f32; hidden_channels * frames];
        let scale = (hidden_channels as f32).sqrt();
        for (t, id) in phoneme_ids.iter().copied().enumerate() {
            if id >= self.embedding.shape[0] {
                return Err(VitsError::InvalidInput(format!(
                    "phoneme id {id} is out of range for {} symbols",
                    self.embedding.shape[0]
                )));
            }
            for c in 0..hidden_channels {
                hidden[c * frames + t] = self.embedding.values[id * hidden_channels + c] * scale;
            }
        }

        for layer_index in 0..self.attention_layers.len() {
            let attention = self.attention_layers[layer_index].infer(&hidden, frames)?;
            add_in_place(&mut hidden, &attention)?;
            self.norm_layers_1[layer_index].apply(&mut hidden, hidden_channels, frames)?;

            let ffn = self.ffn_layers[layer_index].infer(&hidden, frames)?;
            add_in_place(&mut hidden, &ffn)?;
            self.norm_layers_2[layer_index].apply(&mut hidden, hidden_channels, frames)?;
        }

        let stats = conv1d_from_weights(&hidden, frames, &self.projection, 1)?;
        if stats.len() != hidden_channels * 2 * frames {
            return Err(VitsError::InvalidInput(format!(
                "text encoder projection produced {} values, expected {}",
                stats.len(),
                hidden_channels * 2 * frames
            )));
        }
        let split = hidden_channels * frames;
        Ok(TextEncoderOutput {
            hidden,
            prior_mean: stats[..split].to_vec(),
            prior_log_scale: stats[split..].to_vec(),
            channels: hidden_channels,
            frames,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderAttentionLayerWeights {
    pub conv_q: ConvWeights,
    pub conv_k: ConvWeights,
    pub conv_v: ConvWeights,
    pub conv_o: ConvWeights,
    pub emb_rel_k: DenseTensor,
    pub emb_rel_v: DenseTensor,
}

impl TextEncoderAttentionLayerWeights {
    fn infer(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        let hidden_channels = conv_weight_shape(&self.conv_q.weight)?.in_channels;
        let q = conv1d_from_weights(input, frames, &self.conv_q, 1)?;
        let k = conv1d_from_weights(input, frames, &self.conv_k, 1)?;
        let v = conv1d_from_weights(input, frames, &self.conv_v, 1)?;
        let head_dim = *self.emb_rel_k.shape.get(2).ok_or_else(|| {
            VitsError::InvalidInput(format!(
                "relative key embedding {} must have rank 3",
                self.emb_rel_k.name
            ))
        })?;
        if head_dim == 0 || !hidden_channels.is_multiple_of(head_dim) {
            return Err(VitsError::InvalidInput(format!(
                "hidden channel count {hidden_channels} is not divisible by attention head dim {head_dim}"
            )));
        }
        let heads = hidden_channels / head_dim;
        let window = self.emb_rel_k.shape[1] / 2;
        validate_attention_relative("key", &self.emb_rel_k, heads, head_dim, window)?;
        validate_attention_relative("value", &self.emb_rel_v, heads, head_dim, window)?;

        let mut context = vec![0.0f32; hidden_channels * frames];
        let inv_sqrt = 1.0 / (head_dim as f32).sqrt();
        for head in 0..heads {
            for query_t in 0..frames {
                let mut scores = vec![0.0f32; frames];
                for key_t in 0..frames {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        let c = head * head_dim + d;
                        score += q[c * frames + query_t] * k[c * frames + key_t];
                        score += q[c * frames + query_t]
                            * relative_value(
                                &self.emb_rel_k,
                                head,
                                relative_index(query_t, key_t, window),
                                d,
                            );
                    }
                    scores[key_t] = score * inv_sqrt;
                }
                softmax_in_place(&mut scores);
                for d in 0..head_dim {
                    let c = head * head_dim + d;
                    let mut value_sum = 0.0f32;
                    for key_t in 0..frames {
                        let rel = relative_value(
                            &self.emb_rel_v,
                            head,
                            relative_index(query_t, key_t, window),
                            d,
                        );
                        value_sum += scores[key_t] * (v[c * frames + key_t] + rel);
                    }
                    context[c * frames + query_t] = value_sum;
                }
            }
        }
        conv1d_from_weights(&context, frames, &self.conv_o, 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderFfnLayerWeights {
    pub conv_1: ConvWeights,
    pub conv_2: ConvWeights,
}

impl TextEncoderFfnLayerWeights {
    fn infer(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut hidden = conv1d_from_weights(input, frames, &self.conv_1, 1)?;
        relu_in_place(&mut hidden);
        conv1d_from_weights(&hidden, frames, &self.conv_2, 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDurationPredictorWeights {
    pub pre: ConvWeights,
    pub proj: ConvWeights,
    pub dds_convs: DdsConvWeights,
    pub flows: Vec<DurationFlowWeights>,
    pub flow0_m: DenseTensor,
    pub flow0_scale: DenseTensor,
}

impl StochasticDurationPredictorWeights {
    pub fn reverse_log_durations(
        &self,
        text_hidden: &[f32],
        frames: usize,
        noise_scale: f32,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<f32>> {
        let hidden_channels = conv_weight_shape(&self.pre.weight)?.in_channels;
        if text_hidden.len() != hidden_channels * frames {
            return Err(VitsError::InvalidInput(format!(
                "duration predictor input length {} does not match {} channels x {} frames",
                text_hidden.len(),
                hidden_channels,
                frames
            )));
        }

        let mut condition = conv1d_from_weights(text_hidden, frames, &self.pre, 1)?;
        condition = self.dds_convs.infer(&condition, frames, None)?;
        condition = conv1d_from_weights(&condition, frames, &self.proj, 1)?;

        let mut z = vec![0.0f32; 2 * frames];
        for value in &mut z {
            *value = rng.standard_normal() * noise_scale;
        }

        for flow in self.flows.iter().rev() {
            z = flip_channels(&z, 2, frames)?;
            z = flow.reverse(&z, frames, &condition)?;
        }
        reverse_elementwise_affine_in_place(&mut z, 2, frames, &self.flow0_m, &self.flow0_scale)?;
        Ok(z[..frames].to_vec())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DdsConvWeights {
    pub layers: Vec<DdsConvLayerWeights>,
}

impl DdsConvWeights {
    fn infer(
        &self,
        input: &[f32],
        frames: usize,
        conditioning: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if self.layers.is_empty() {
            return Err(VitsError::InvalidInput(
                "DDSConv must have at least one layer".to_string(),
            ));
        }
        let channels = conv_weight_shape(&self.layers[0].conv_sep.weight)?.out_channels;
        if input.len() != channels * frames {
            return Err(VitsError::InvalidInput(format!(
                "DDSConv input length {} does not match {} channels x {} frames",
                input.len(),
                channels,
                frames
            )));
        }
        let mut current = input.to_vec();
        if let Some(conditioning) = conditioning {
            add_in_place(&mut current, conditioning)?;
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let dilation = 3usize.pow(index as u32);
            let mut hidden = depthwise_conv1d_from_weights(
                &current,
                frames,
                &layer.conv_sep,
                channels,
                dilation,
            )?;
            layer.norm_1.apply(&mut hidden, channels, frames)?;
            gelu_in_place(&mut hidden);
            hidden = conv1d_from_weights(&hidden, frames, &layer.conv_1x1, 1)?;
            layer.norm_2.apply(&mut hidden, channels, frames)?;
            gelu_in_place(&mut hidden);
            add_in_place(&mut current, &hidden)?;
        }
        Ok(current)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DdsConvLayerWeights {
    pub conv_sep: ConvWeights,
    pub conv_1x1: ConvWeights,
    pub norm_1: LayerNormWeights,
    pub norm_2: LayerNormWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DurationFlowWeights {
    pub index: usize,
    pub pre: ConvWeights,
    pub proj: ConvWeights,
    pub dds_convs: DdsConvWeights,
}

impl DurationFlowWeights {
    fn reverse(&self, latent: &[f32], frames: usize, conditioning: &[f32]) -> Result<Vec<f32>> {
        if latent.len() != 2 * frames {
            return Err(VitsError::InvalidInput(format!(
                "duration flow latent length {} does not match 2 channels x {} frames",
                latent.len(),
                frames
            )));
        }
        let x0 = latent[..frames].to_vec();
        let x1 = &latent[frames..];
        let mut hidden = conv1d_from_weights(&x0, frames, &self.pre, 1)?;
        hidden = self.dds_convs.infer(&hidden, frames, Some(conditioning))?;
        let params = conv1d_from_weights(&hidden, frames, &self.proj, 1)?;
        if params.len() != 29 * frames {
            return Err(VitsError::InvalidInput(format!(
                "duration flow {} projection produced {} values, expected {}",
                self.index,
                params.len(),
                29 * frames
            )));
        }

        let mut out = latent.to_vec();
        for t in 0..frames {
            let mut widths = [0.0f32; 10];
            let mut heights = [0.0f32; 10];
            let mut derivatives = [0.0f32; 9];
            for i in 0..10 {
                widths[i] = params[i * frames + t] / 192.0f32.sqrt();
                heights[i] = params[(10 + i) * frames + t] / 192.0f32.sqrt();
            }
            for i in 0..9 {
                derivatives[i] = params[(20 + i) * frames + t];
            }
            out[frames + t] =
                rational_quadratic_spline_inverse(x1[t], &widths, &heights, &derivatives);
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidualCouplingFlowWeights {
    pub blocks: Vec<ResidualCouplingBlockWeights>,
}

impl ResidualCouplingFlowWeights {
    pub fn reverse(&self, latent: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
        if latent.len() != channels * frames {
            return Err(VitsError::InvalidInput(format!(
                "flow latent length {} does not match {} channels x {} frames",
                latent.len(),
                channels,
                frames
            )));
        }
        let mut current = latent.to_vec();
        for block in self.blocks.iter().rev() {
            current = flip_channels(&current, channels, frames)?;
            current = block.reverse(&current, channels, frames)?;
        }
        Ok(current)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidualCouplingBlockWeights {
    pub index: usize,
    pub pre: ConvWeights,
    pub post: ConvWeights,
    pub wavenet: WavenetWeights,
}

impl ResidualCouplingBlockWeights {
    fn reverse(&self, latent: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
        if !channels.is_multiple_of(2) {
            return Err(VitsError::InvalidInput(format!(
                "residual coupling channels {channels} must be even"
            )));
        }
        let half_channels = channels / 2;
        let first_half = latent[..half_channels * frames].to_vec();
        let hidden = conv1d_from_weights(&first_half, frames, &self.pre, 1)?;
        let shift = self.wavenet.infer(&hidden, frames)?;
        let shift = conv1d_from_weights(&shift, frames, &self.post, 1)?;
        residual_coupling_reverse(latent, channels, frames, &shift, None)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WavenetWeights {
    pub in_layers: Vec<ConvWeights>,
    pub res_skip_layers: Vec<ConvWeights>,
}

impl WavenetWeights {
    fn infer(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        if self.in_layers.len() != self.res_skip_layers.len() {
            return Err(VitsError::InvalidInput(format!(
                "WaveNet has mismatched layer counts: in_layers={}, res_skip_layers={}",
                self.in_layers.len(),
                self.res_skip_layers.len()
            )));
        }
        if self.in_layers.is_empty() {
            return Err(VitsError::InvalidInput(
                "WaveNet must have at least one layer".to_string(),
            ));
        }
        let hidden_channels = conv_weight_shape(&self.in_layers[0].weight)?.in_channels;
        let mut current = input.to_vec();
        let mut output = vec![0.0f32; hidden_channels * frames];
        for (index, (in_layer, res_skip_layer)) in
            self.in_layers.iter().zip(&self.res_skip_layers).enumerate()
        {
            let acts = conv1d_from_weights(&current, frames, in_layer, 1)?;
            let acts = gated_tanh_sigmoid(&acts, hidden_channels, frames)?;
            let res_skip = conv1d_from_weights(&acts, frames, res_skip_layer, 1)?;
            if index + 1 == self.in_layers.len() {
                add_in_place(&mut output, &res_skip)?;
            } else {
                let split = hidden_channels * frames;
                add_in_place(&mut current, &res_skip[..split])?;
                add_in_place(&mut output, &res_skip[split..])?;
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorWeights {
    pub conv_pre: ConvWeights,
    pub upsample_layers: Vec<ConvWeights>,
    pub resblocks: Vec<GeneratorResBlockWeights>,
    pub conv_post: ConvWeights,
}

impl GeneratorWeights {
    pub fn infer(&self, latent: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut current = conv1d_from_weights(latent, frames, &self.conv_pre, 1)?;
        let mut current_frames = conv_output_len(frames, &self.conv_pre, 1)?;
        let resblocks_per_upsample = self
            .resblocks
            .len()
            .checked_div(self.upsample_layers.len())
            .ok_or_else(|| {
                VitsError::InvalidInput("generator must have upsample layers".to_string())
            })?;
        if resblocks_per_upsample == 0
            || resblocks_per_upsample * self.upsample_layers.len() != self.resblocks.len()
        {
            return Err(VitsError::InvalidInput(format!(
                "generator has {} resblocks for {} upsample layers",
                self.resblocks.len(),
                self.upsample_layers.len()
            )));
        }

        for (up_index, upsample) in self.upsample_layers.iter().enumerate() {
            leaky_relu_in_place(&mut current, 0.1);
            current = conv_transpose1d_from_weights(&current, current_frames, upsample)?;
            current_frames = conv_transpose_output_len(current_frames, upsample)?;

            let group_start = up_index * resblocks_per_upsample;
            let group = &self.resblocks[group_start..group_start + resblocks_per_upsample];
            let mut accumulated: Option<Vec<f32>> = None;
            for resblock in group {
                let out = resblock.infer(&current, current_frames)?;
                if let Some(accumulated) = &mut accumulated {
                    for (dst, value) in accumulated.iter_mut().zip(out) {
                        *dst += value;
                    }
                } else {
                    accumulated = Some(out);
                }
            }
            current = accumulated.expect("resblock group is non-empty");
            let scale = 1.0 / group.len() as f32;
            for value in &mut current {
                *value *= scale;
            }
        }

        leaky_relu_in_place(&mut current, 0.1);
        let mut audio = conv1d_from_weights(&current, current_frames, &self.conv_post, 1)?;
        for sample in &mut audio {
            *sample = sample.tanh();
        }
        Ok(audio)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorResBlockWeights {
    pub index: usize,
    pub convs: Vec<ConvWeights>,
}

impl GeneratorResBlockWeights {
    fn infer(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut current = input.to_vec();
        for (conv_index, conv) in self.convs.iter().enumerate() {
            let dilation = if conv_index == 0 { 1 } else { 3 };
            let mut hidden = current.clone();
            leaky_relu_in_place(&mut hidden, 0.1);
            hidden = conv1d_from_weights(&hidden, frames, conv, dilation)?;
            if hidden.len() != current.len() {
                return Err(VitsError::InvalidInput(format!(
                    "generator resblock {} conv {} changed length from {} to {}",
                    self.index,
                    conv_index,
                    current.len(),
                    hidden.len()
                )));
            }
            for (dst, value) in current.iter_mut().zip(hidden) {
                *dst += value;
            }
        }
        Ok(current)
    }
}

impl VitsWeights {
    pub fn from_onnx_file(path: &Path, config: VitsWeightConfig) -> Result<Self> {
        let store = load_onnx_initializers(path).map_err(map_onnx_error)?;
        Self::from_initializers(&store, config)
    }

    pub fn from_initializers(
        store: &OnnxInitializerStore,
        config: VitsWeightConfig,
    ) -> Result<Self> {
        if config.num_symbols == 0 {
            return Err(VitsError::InvalidInput(
                "num_symbols must be greater than zero".to_string(),
            ));
        }
        if config.num_speakers == 0 {
            return Err(VitsError::InvalidInput(
                "num_speakers must be greater than zero".to_string(),
            ));
        }

        let embedding = load_text_embedding(store, config.num_symbols)?;
        let hidden_channels = *embedding.shape.get(1).ok_or_else(|| {
            VitsError::InvalidInput(format!(
                "text embedding {} must have rank 2, got {:?}",
                embedding.name, embedding.shape
            ))
        })?;
        expect_shape(
            &embedding.name,
            &embedding.shape,
            &[config.num_symbols, hidden_channels],
        )?;
        let half_channels = hidden_channels / 2;
        if half_channels * 2 != hidden_channels {
            return Err(VitsError::InvalidInput(format!(
                "hidden channel count {hidden_channels} must be even"
            )));
        }

        let text_encoder = load_text_encoder(store, embedding, hidden_channels)?;
        let deterministic_duration_predictor =
            load_optional_deterministic_duration_predictor(store, hidden_channels)?;
        let duration_predictor = load_stochastic_duration_predictor(store, hidden_channels)?;
        let residual_coupling_flow = load_residual_coupling_flow(store, hidden_channels)?;
        let generator = load_generator(store, hidden_channels)?;
        let speaker_embedding =
            load_speaker_embedding(store, config.num_speakers, hidden_channels)?;

        Ok(Self {
            hidden_channels,
            text_encoder,
            deterministic_duration_predictor,
            duration_predictor,
            residual_coupling_flow,
            generator,
            speaker_embedding,
        })
    }
}

fn load_text_encoder(
    store: &OnnxInitializerStore,
    embedding: DenseTensor,
    hidden_channels: usize,
) -> Result<TextEncoderWeights> {
    let projection = load_conv(
        store,
        "enc_p.proj",
        &[hidden_channels * 2, hidden_channels, 1],
        Some(&[hidden_channels * 2]),
    )?;
    let mut attention_layers = Vec::new();
    let mut ffn_layers = Vec::new();
    let mut norm_layers_1 = Vec::new();
    let mut norm_layers_2 = Vec::new();
    for index in sequence_indices(store, "enc_p.encoder.attn_layers.", ".conv_q.weight") {
        let prefix = format!("enc_p.encoder.attn_layers.{index}");
        attention_layers.push(TextEncoderAttentionLayerWeights {
            conv_q: load_conv(
                store,
                &format!("{prefix}.conv_q"),
                &[hidden_channels, hidden_channels, 1],
                Some(&[hidden_channels]),
            )?,
            conv_k: load_conv(
                store,
                &format!("{prefix}.conv_k"),
                &[hidden_channels, hidden_channels, 1],
                Some(&[hidden_channels]),
            )?,
            conv_v: load_conv(
                store,
                &format!("{prefix}.conv_v"),
                &[hidden_channels, hidden_channels, 1],
                Some(&[hidden_channels]),
            )?,
            conv_o: load_conv(
                store,
                &format!("{prefix}.conv_o"),
                &[hidden_channels, hidden_channels, 1],
                Some(&[hidden_channels]),
            )?,
            emb_rel_k: load_rel_embedding(store, &format!("{prefix}.emb_rel_k"), hidden_channels)?,
            emb_rel_v: load_rel_embedding(store, &format!("{prefix}.emb_rel_v"), hidden_channels)?,
        });

        let ffn_prefix = format!("enc_p.encoder.ffn_layers.{index}");
        let filter_channels = hidden_channels * 4;
        ffn_layers.push(TextEncoderFfnLayerWeights {
            conv_1: load_conv(
                store,
                &format!("{ffn_prefix}.conv_1"),
                &[filter_channels, hidden_channels, 3],
                Some(&[filter_channels]),
            )?,
            conv_2: load_conv(
                store,
                &format!("{ffn_prefix}.conv_2"),
                &[hidden_channels, filter_channels, 3],
                Some(&[hidden_channels]),
            )?,
        });

        norm_layers_1.push(load_layer_norm(
            store,
            &format!("enc_p.encoder.norm_layers_1.{index}"),
            hidden_channels,
        )?);
        norm_layers_2.push(load_layer_norm(
            store,
            &format!("enc_p.encoder.norm_layers_2.{index}"),
            hidden_channels,
        )?);
    }
    if attention_layers.is_empty() {
        return Err(VitsError::InvalidInput(
            "no text encoder attention layers found".to_string(),
        ));
    }
    Ok(TextEncoderWeights {
        embedding,
        projection,
        attention_layers,
        ffn_layers,
        norm_layers_1,
        norm_layers_2,
    })
}

fn load_optional_deterministic_duration_predictor(
    store: &OnnxInitializerStore,
    hidden_channels: usize,
) -> Result<Option<DeterministicDurationPredictorWeights>> {
    if store.get("dp.conv_1.weight").is_none() {
        return Ok(None);
    }
    let conv1_weight = load_ranked_f32(store, "dp.conv_1.weight", 3)?;
    if conv1_weight.shape[1] != hidden_channels {
        return Err(wrong_shape(
            &conv1_weight.name,
            &conv1_weight.shape,
            &[
                conv1_weight.shape[0],
                hidden_channels,
                conv1_weight.shape[2],
            ],
        ));
    }
    let filter_channels = conv1_weight.shape[0];
    let kernel_size = conv1_weight.shape[2];
    Ok(Some(DeterministicDurationPredictorWeights {
        channels: hidden_channels,
        filter_channels,
        kernel_size,
        conv1_weight: conv1_weight.values,
        conv1_bias: load_f32(store, "dp.conv_1.bias", &[filter_channels])?.values,
        norm1_gamma: load_f32(store, "dp.norm_1.gamma", &[filter_channels])?.values,
        norm1_beta: load_f32(store, "dp.norm_1.beta", &[filter_channels])?.values,
        conv2_weight: load_f32(
            store,
            "dp.conv_2.weight",
            &[filter_channels, filter_channels, kernel_size],
        )?
        .values,
        conv2_bias: load_f32(store, "dp.conv_2.bias", &[filter_channels])?.values,
        norm2_gamma: load_f32(store, "dp.norm_2.gamma", &[filter_channels])?.values,
        norm2_beta: load_f32(store, "dp.norm_2.beta", &[filter_channels])?.values,
        proj_weight: load_f32(store, "dp.proj.weight", &[1, filter_channels, 1])?.values,
        proj_bias: load_f32(store, "dp.proj.bias", &[1])?.values[0],
    }))
}

fn load_stochastic_duration_predictor(
    store: &OnnxInitializerStore,
    hidden_channels: usize,
) -> Result<StochasticDurationPredictorWeights> {
    Ok(StochasticDurationPredictorWeights {
        pre: load_conv(
            store,
            "dp.pre",
            &[hidden_channels, hidden_channels, 1],
            Some(&[hidden_channels]),
        )?,
        proj: load_conv(
            store,
            "dp.proj",
            &[hidden_channels, hidden_channels, 1],
            Some(&[hidden_channels]),
        )?,
        dds_convs: load_dds_convs(store, "dp.convs", hidden_channels)?,
        flows: [3, 5, 7]
            .into_iter()
            .map(|index| load_duration_flow(store, index, hidden_channels))
            .collect::<Result<Vec<_>>>()?,
        flow0_m: load_f32(store, "dp.flows.0.m", &[2, 1])?,
        flow0_scale: load_f32(store, "/dp/flows.0/Exp_output_0", &[2, 1])?,
    })
}

fn load_duration_flow(
    store: &OnnxInitializerStore,
    index: usize,
    hidden_channels: usize,
) -> Result<DurationFlowWeights> {
    let prefix = format!("dp.flows.{index}");
    let proj_weight = load_ranked_f32(store, &format!("{prefix}.proj.weight"), 3)?;
    if proj_weight.shape[1] != hidden_channels || proj_weight.shape[2] != 1 {
        return Err(wrong_shape(
            &proj_weight.name,
            &proj_weight.shape,
            &[proj_weight.shape[0], hidden_channels, 1],
        ));
    }
    let proj_bias = load_f32(
        store,
        &format!("{prefix}.proj.bias"),
        &[proj_weight.shape[0]],
    )?;

    Ok(DurationFlowWeights {
        index,
        pre: load_conv(
            store,
            &format!("{prefix}.pre"),
            &[hidden_channels, 1, 1],
            Some(&[hidden_channels]),
        )?,
        proj: ConvWeights {
            weight: proj_weight,
            bias: Some(proj_bias),
        },
        dds_convs: load_dds_convs(store, &format!("{prefix}.convs"), hidden_channels)?,
    })
}

fn load_dds_convs(
    store: &OnnxInitializerStore,
    prefix: &str,
    hidden_channels: usize,
) -> Result<DdsConvWeights> {
    let mut layers = Vec::new();
    for index in sequence_indices(store, &format!("{prefix}.convs_sep."), ".weight") {
        layers.push(DdsConvLayerWeights {
            conv_sep: load_conv(
                store,
                &format!("{prefix}.convs_sep.{index}"),
                &[hidden_channels, 1, 3],
                Some(&[hidden_channels]),
            )?,
            conv_1x1: load_conv(
                store,
                &format!("{prefix}.convs_1x1.{index}"),
                &[hidden_channels, hidden_channels, 1],
                Some(&[hidden_channels]),
            )?,
            norm_1: load_layer_norm(store, &format!("{prefix}.norms_1.{index}"), hidden_channels)?,
            norm_2: load_layer_norm(store, &format!("{prefix}.norms_2.{index}"), hidden_channels)?,
        });
    }
    if layers.is_empty() {
        return Err(VitsError::InvalidInput(format!(
            "no DDS conv layers found under {prefix}"
        )));
    }
    Ok(DdsConvWeights { layers })
}

fn load_residual_coupling_flow(
    store: &OnnxInitializerStore,
    hidden_channels: usize,
) -> Result<ResidualCouplingFlowWeights> {
    let half_channels = hidden_channels / 2;
    let anonymous_weight_names = sequence_indices(store, "onnx::Conv_", "")
        .into_iter()
        .map(|suffix| format!("onnx::Conv_{suffix}"))
        .collect::<Vec<_>>();
    let flow_indices = sequence_indices(store, "flow.flows.", ".pre.weight");
    let expected_anonymous = flow_indices.len() * 8;
    if anonymous_weight_names.len() < expected_anonymous {
        return Err(VitsError::InvalidInput(format!(
            "residual coupling flow needs {expected_anonymous} anonymous WaveNet weights, found {}",
            anonymous_weight_names.len()
        )));
    }

    let mut anonymous_offset = 0;
    let mut blocks = Vec::new();
    for index in flow_indices {
        let prefix = format!("flow.flows.{index}");
        let mut in_layers = Vec::new();
        let mut res_skip_layers = Vec::new();
        for layer in 0..4 {
            let in_weight = &anonymous_weight_names[anonymous_offset];
            anonymous_offset += 1;
            in_layers.push(ConvWeights {
                weight: load_f32(store, in_weight, &[hidden_channels * 2, hidden_channels, 5])?,
                bias: Some(load_f32(
                    store,
                    &format!("{prefix}.enc.in_layers.{layer}.bias"),
                    &[hidden_channels * 2],
                )?),
            });

            let res_weight = &anonymous_weight_names[anonymous_offset];
            anonymous_offset += 1;
            let res_out = if layer == 3 {
                hidden_channels
            } else {
                hidden_channels * 2
            };
            res_skip_layers.push(ConvWeights {
                weight: load_f32(store, res_weight, &[res_out, hidden_channels, 1])?,
                bias: Some(load_f32(
                    store,
                    &format!("{prefix}.enc.res_skip_layers.{layer}.bias"),
                    &[res_out],
                )?),
            });
        }
        blocks.push(ResidualCouplingBlockWeights {
            index,
            pre: load_conv(
                store,
                &format!("{prefix}.pre"),
                &[hidden_channels, half_channels, 1],
                Some(&[hidden_channels]),
            )?,
            post: load_conv(
                store,
                &format!("{prefix}.post"),
                &[half_channels, hidden_channels, 1],
                Some(&[half_channels]),
            )?,
            wavenet: WavenetWeights {
                in_layers,
                res_skip_layers,
            },
        });
    }
    if blocks.is_empty() {
        return Err(VitsError::InvalidInput(
            "no residual coupling flow blocks found".to_string(),
        ));
    }
    Ok(ResidualCouplingFlowWeights { blocks })
}

fn load_generator(
    store: &OnnxInitializerStore,
    hidden_channels: usize,
) -> Result<GeneratorWeights> {
    let conv_pre_weight = load_ranked_f32(store, "dec.conv_pre.weight", 3)?;
    if conv_pre_weight.shape[1] != hidden_channels || conv_pre_weight.shape[2] != 7 {
        return Err(wrong_shape(
            &conv_pre_weight.name,
            &conv_pre_weight.shape,
            &[conv_pre_weight.shape[0], hidden_channels, 7],
        ));
    }
    let mut channels = conv_pre_weight.shape[0];
    let conv_pre = ConvWeights {
        bias: Some(load_f32(store, "dec.conv_pre.bias", &[channels])?),
        weight: conv_pre_weight,
    };

    let mut upsample_layers = Vec::new();
    for index in sequence_indices(store, "dec.ups.", ".weight") {
        let weight = load_ranked_f32(store, &format!("dec.ups.{index}.weight"), 3)?;
        if weight.shape[0] != channels {
            return Err(wrong_shape(
                &weight.name,
                &weight.shape,
                &[channels, weight.shape[1], weight.shape[2]],
            ));
        }
        channels = weight.shape[1];
        upsample_layers.push(ConvWeights {
            bias: Some(load_f32(
                store,
                &format!("dec.ups.{index}.bias"),
                &[channels],
            )?),
            weight,
        });
    }
    if upsample_layers.is_empty() {
        return Err(VitsError::InvalidInput(
            "no generator upsample layers found".to_string(),
        ));
    }

    let mut resblocks = Vec::new();
    for index in sequence_indices(store, "dec.resblocks.", ".convs.0.weight") {
        let mut convs = Vec::new();
        for conv_index in 0..=1 {
            let weight = load_ranked_f32(
                store,
                &format!("dec.resblocks.{index}.convs.{conv_index}.weight"),
                3,
            )?;
            if weight.shape[0] != weight.shape[1] {
                return Err(VitsError::InvalidInput(format!(
                    "resblock conv {} must have matching in/out channels, got {:?}",
                    weight.name, weight.shape
                )));
            }
            convs.push(ConvWeights {
                bias: Some(load_f32(
                    store,
                    &format!("dec.resblocks.{index}.convs.{conv_index}.bias"),
                    &[weight.shape[0]],
                )?),
                weight,
            });
        }
        resblocks.push(GeneratorResBlockWeights { index, convs });
    }
    if resblocks.is_empty() {
        return Err(VitsError::InvalidInput(
            "no generator residual blocks found".to_string(),
        ));
    }

    let conv_post_weight = load_f32(store, "dec.conv_post.weight", &[1, channels, 7])?;
    Ok(GeneratorWeights {
        conv_pre,
        upsample_layers,
        resblocks,
        conv_post: ConvWeights {
            weight: conv_post_weight,
            bias: load_optional_f32(store, "dec.conv_post.bias", &[1])?,
        },
    })
}

fn load_speaker_embedding(
    store: &OnnxInitializerStore,
    num_speakers: usize,
    hidden_channels: usize,
) -> Result<Option<DenseTensor>> {
    if num_speakers == 1 {
        return Ok(None);
    }
    Ok(Some(load_any_f32(
        store,
        &["emb_g.weight", "speaker_embedding.weight"],
        &[num_speakers, hidden_channels],
    )?))
}

fn load_text_embedding(store: &OnnxInitializerStore, num_symbols: usize) -> Result<DenseTensor> {
    for name in ["sid", "enc_p.emb.weight"] {
        if store.get(name).is_some() {
            let tensor = load_ranked_f32(store, name, 2)?;
            if tensor.shape[0] != num_symbols {
                return Err(wrong_shape(
                    name,
                    &tensor.shape,
                    &[num_symbols, tensor.shape[1]],
                ));
            }
            return Ok(tensor);
        }
    }
    Err(VitsError::InvalidInput(
        "missing text embedding tensor; tried sid, enc_p.emb.weight".to_string(),
    ))
}

fn load_rel_embedding(
    store: &OnnxInitializerStore,
    name: &str,
    hidden_channels: usize,
) -> Result<DenseTensor> {
    let tensor = load_ranked_f32(store, name, 3)?;
    if tensor.shape[0] != 1 || tensor.shape[2] != hidden_channels / 2 {
        return Err(wrong_shape(
            name,
            &tensor.shape,
            &[1, tensor.shape[1], hidden_channels / 2],
        ));
    }
    Ok(tensor)
}

fn load_conv(
    store: &OnnxInitializerStore,
    prefix: &str,
    weight_shape: &[usize],
    bias_shape: Option<&[usize]>,
) -> Result<ConvWeights> {
    Ok(ConvWeights {
        weight: load_f32(store, &format!("{prefix}.weight"), weight_shape)?,
        bias: match bias_shape {
            Some(shape) => Some(load_f32(store, &format!("{prefix}.bias"), shape)?),
            None => None,
        },
    })
}

fn conv1d_from_weights(
    input: &[f32],
    frames: usize,
    weights: &ConvWeights,
    dilation: usize,
) -> Result<Vec<f32>> {
    let shape = conv_weight_shape(&weights.weight)?;
    conv1d(
        input,
        frames,
        Conv1dParams {
            in_channels: shape.in_channels,
            out_channels: shape.out_channels,
            kernel_size: shape.kernel_size,
            stride: 1,
            padding: same_padding(shape.kernel_size, dilation),
            dilation,
            groups: 1,
        },
        &weights.weight.values,
        weights.bias.as_ref().map(|bias| bias.values.as_slice()),
    )
}

fn depthwise_conv1d_from_weights(
    input: &[f32],
    frames: usize,
    weights: &ConvWeights,
    channels: usize,
    dilation: usize,
) -> Result<Vec<f32>> {
    let shape = conv_weight_shape(&weights.weight)?;
    if shape.out_channels != channels || shape.in_channels != 1 {
        return Err(VitsError::InvalidInput(format!(
            "depthwise conv tensor {} shape {:?} is incompatible with {} channels",
            weights.weight.name, weights.weight.shape, channels
        )));
    }
    conv1d(
        input,
        frames,
        Conv1dParams {
            in_channels: channels,
            out_channels: channels,
            kernel_size: shape.kernel_size,
            stride: 1,
            padding: same_padding(shape.kernel_size, dilation),
            dilation,
            groups: channels,
        },
        &weights.weight.values,
        weights.bias.as_ref().map(|bias| bias.values.as_slice()),
    )
}

fn embedding_hidden_channels(embedding: &DenseTensor) -> Result<usize> {
    if embedding.shape.len() != 2 {
        return Err(VitsError::InvalidInput(format!(
            "embedding tensor {} must have rank 2, got {:?}",
            embedding.name, embedding.shape
        )));
    }
    Ok(embedding.shape[1])
}

fn expand_channel_major_by_durations(
    values: &[f32],
    channels: usize,
    durations: &[usize],
) -> Result<Vec<f32>> {
    expand_by_durations(values, channels, durations)
}

fn add_in_place(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(VitsError::InvalidInput(format!(
            "residual add length mismatch: {} vs {}",
            dst.len(),
            src.len()
        )));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += src;
    }
    Ok(())
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn gelu_in_place(values: &mut [f32]) {
    for value in values {
        let x = *value;
        *value = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / values.len().max(1) as f32;
        values.fill(uniform);
        return;
    }
    for value in values {
        *value /= sum;
    }
}

fn reverse_elementwise_affine_in_place(
    values: &mut [f32],
    channels: usize,
    frames: usize,
    bias: &DenseTensor,
    scale: &DenseTensor,
) -> Result<()> {
    expect_shape(&bias.name, &bias.shape, &[channels, 1])?;
    expect_shape(&scale.name, &scale.shape, &[channels, 1])?;
    if values.len() != channels * frames {
        return Err(VitsError::InvalidInput(format!(
            "affine latent length {} does not match {} channels x {} frames",
            values.len(),
            channels,
            frames
        )));
    }
    for c in 0..channels {
        let b = bias.values[c];
        let s = scale.values[c];
        if s == 0.0 {
            return Err(VitsError::InvalidInput(format!(
                "duration affine scale for channel {c} is zero"
            )));
        }
        for t in 0..frames {
            values[c * frames + t] = (values[c * frames + t] - b) / s;
        }
    }
    Ok(())
}

fn rational_quadratic_spline_inverse(
    value: f32,
    unnormalized_widths: &[f32; 10],
    unnormalized_heights: &[f32; 10],
    unnormalized_derivatives: &[f32; 9],
) -> f32 {
    const NUM_BINS: usize = 10;
    const TAIL_BOUND: f32 = 5.0;
    const MIN_BIN_WIDTH: f32 = 1e-3;
    const MIN_BIN_HEIGHT: f32 = 1e-3;
    const MIN_DERIVATIVE: f32 = 1e-3;

    if value <= -TAIL_BOUND || value >= TAIL_BOUND {
        return value;
    }

    let widths = constrained_bins::<NUM_BINS>(unnormalized_widths, MIN_BIN_WIDTH, TAIL_BOUND * 2.0);
    let heights =
        constrained_bins::<NUM_BINS>(unnormalized_heights, MIN_BIN_HEIGHT, TAIL_BOUND * 2.0);
    let derivatives = constrained_derivatives(unnormalized_derivatives, MIN_DERIVATIVE);

    let mut cumwidths = [0.0f32; NUM_BINS + 1];
    let mut cumheights = [0.0f32; NUM_BINS + 1];
    cumwidths[0] = -TAIL_BOUND;
    cumheights[0] = -TAIL_BOUND;
    for i in 0..NUM_BINS {
        cumwidths[i + 1] = cumwidths[i] + widths[i];
        cumheights[i + 1] = cumheights[i] + heights[i];
    }
    cumwidths[NUM_BINS] = TAIL_BOUND;
    cumheights[NUM_BINS] = TAIL_BOUND;

    let bin = search_sorted_bin(&cumheights, value).min(NUM_BINS - 1);
    let input_delta = value - cumheights[bin];
    let width = widths[bin];
    let height = heights[bin];
    let delta = height / width;
    let derivative_left = derivatives[bin];
    let derivative_right = derivatives[bin + 1];
    let derivative_sum = derivative_left + derivative_right - 2.0 * delta;

    let a = input_delta * derivative_sum + height * (delta - derivative_left);
    let b = height * derivative_left - input_delta * derivative_sum;
    let c = -delta * input_delta;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let root = if a.abs() < 1e-7 {
        (-c / b).clamp(0.0, 1.0)
    } else {
        (2.0 * c / (-b - discriminant.sqrt())).clamp(0.0, 1.0)
    };
    cumwidths[bin] + root * width
}

fn constrained_bins<const N: usize>(unnormalized: &[f32; N], min_bin: f32, total: f32) -> [f32; N] {
    let mut probs = *unnormalized;
    softmax_in_place(&mut probs);
    let available = total - min_bin * N as f32;
    for value in &mut probs {
        *value = min_bin + available * *value;
    }
    probs
}

fn constrained_derivatives(unnormalized: &[f32; 9], min_derivative: f32) -> [f32; 11] {
    let mut derivatives = [0.0f32; 11];
    derivatives[0] = 1.0;
    derivatives[10] = 1.0;
    for (dst, src) in derivatives[1..10].iter_mut().zip(unnormalized) {
        *dst = min_derivative + softplus(*src);
    }
    derivatives
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn search_sorted_bin(cumulative: &[f32], value: f32) -> usize {
    cumulative
        .windows(2)
        .position(|pair| value >= pair[0] && value <= pair[1])
        .unwrap_or_else(|| cumulative.len().saturating_sub(2))
}

fn validate_attention_relative(
    label: &str,
    tensor: &DenseTensor,
    heads: usize,
    head_dim: usize,
    window: usize,
) -> Result<()> {
    if tensor.shape.len() != 3
        || (tensor.shape[0] != 1 && tensor.shape[0] != heads)
        || tensor.shape[1] != window * 2 + 1
        || tensor.shape[2] != head_dim
    {
        return Err(VitsError::InvalidInput(format!(
            "relative attention {label} tensor {} shape {:?} is incompatible with heads={}, window={}, head_dim={}",
            tensor.name, tensor.shape, heads, window, head_dim
        )));
    }
    validate_storage_len(&tensor.name, &tensor.shape, tensor.values.len())
}

fn relative_index(query_t: usize, key_t: usize, window: usize) -> usize {
    let raw = key_t as isize - query_t as isize + window as isize;
    raw.clamp(0, (window * 2) as isize) as usize
}

fn relative_value(tensor: &DenseTensor, head: usize, relative_index: usize, dim: usize) -> f32 {
    let rel_heads = tensor.shape[0];
    let rel_head = if rel_heads == 1 { 0 } else { head };
    let rel_positions = tensor.shape[1];
    let head_dim = tensor.shape[2];
    tensor.values[(rel_head * rel_positions + relative_index) * head_dim + dim]
}

fn conv_output_len(frames: usize, weights: &ConvWeights, dilation: usize) -> Result<usize> {
    let shape = conv_weight_shape(&weights.weight)?;
    Conv1dParams {
        in_channels: shape.in_channels,
        out_channels: shape.out_channels,
        kernel_size: shape.kernel_size,
        stride: 1,
        padding: same_padding(shape.kernel_size, dilation),
        dilation,
        groups: 1,
    }
    .output_len(frames)
}

fn conv_transpose1d_from_weights(
    input: &[f32],
    frames: usize,
    weights: &ConvWeights,
) -> Result<Vec<f32>> {
    let shape = conv_transpose_weight_shape(&weights.weight)?;
    let stride = shape.kernel_size / 2;
    conv_transpose1d(
        input,
        frames,
        ConvTranspose1dParams {
            in_channels: shape.in_channels,
            out_channels: shape.out_channels,
            kernel_size: shape.kernel_size,
            stride,
            padding: (shape.kernel_size - stride) / 2,
            dilation: 1,
            groups: 1,
            output_padding: 0,
        },
        &weights.weight.values,
        weights.bias.as_ref().map(|bias| bias.values.as_slice()),
    )
}

fn conv_transpose_output_len(frames: usize, weights: &ConvWeights) -> Result<usize> {
    let shape = conv_transpose_weight_shape(&weights.weight)?;
    let stride = shape.kernel_size / 2;
    ConvTranspose1dParams {
        in_channels: shape.in_channels,
        out_channels: shape.out_channels,
        kernel_size: shape.kernel_size,
        stride,
        padding: (shape.kernel_size - stride) / 2,
        dilation: 1,
        groups: 1,
        output_padding: 0,
    }
    .output_len(frames)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConvShape {
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
}

fn conv_weight_shape(weight: &DenseTensor) -> Result<ConvShape> {
    if weight.shape.len() != 3 {
        return Err(VitsError::InvalidInput(format!(
            "conv tensor {} must have rank 3, got {:?}",
            weight.name, weight.shape
        )));
    }
    Ok(ConvShape {
        out_channels: weight.shape[0],
        in_channels: weight.shape[1],
        kernel_size: weight.shape[2],
    })
}

fn conv_transpose_weight_shape(weight: &DenseTensor) -> Result<ConvShape> {
    if weight.shape.len() != 3 {
        return Err(VitsError::InvalidInput(format!(
            "conv-transpose tensor {} must have rank 3, got {:?}",
            weight.name, weight.shape
        )));
    }
    Ok(ConvShape {
        in_channels: weight.shape[0],
        out_channels: weight.shape[1],
        kernel_size: weight.shape[2],
    })
}

fn load_layer_norm(
    store: &OnnxInitializerStore,
    prefix: &str,
    hidden_channels: usize,
) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        gamma: load_f32(store, &format!("{prefix}.gamma"), &[hidden_channels])?,
        beta: load_f32(store, &format!("{prefix}.beta"), &[hidden_channels])?,
    })
}

fn load_any_f32(
    store: &OnnxInitializerStore,
    names: &[&str],
    expected_shape: &[usize],
) -> Result<DenseTensor> {
    for name in names {
        if store.get(name).is_some() {
            return load_f32(store, name, expected_shape);
        }
    }
    Err(VitsError::InvalidInput(format!(
        "missing tensor; tried {}",
        names.join(", ")
    )))
}

fn load_optional_f32(
    store: &OnnxInitializerStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<Option<DenseTensor>> {
    if store.get(name).is_some() {
        Ok(Some(load_f32(store, name, expected_shape)?))
    } else {
        Ok(None)
    }
}

fn load_ranked_f32(store: &OnnxInitializerStore, name: &str, rank: usize) -> Result<DenseTensor> {
    let tensor = store.required(name).map_err(map_onnx_error)?;
    if tensor.data_type != OnnxTensorType::Float32 {
        return Err(VitsError::InvalidInput(format!(
            "tensor {name} has dtype {:?}, expected FLOAT",
            tensor.data_type
        )));
    }
    if tensor.dims.len() != rank {
        return Err(VitsError::InvalidInput(format!(
            "tensor {name} has rank {}, expected {rank}; shape {:?}",
            tensor.dims.len(),
            tensor.dims
        )));
    }
    let values = tensor.f32_values().map_err(map_onnx_error)?;
    validate_storage_len(name, &tensor.dims, values.len())?;
    Ok(DenseTensor {
        name: name.to_string(),
        shape: tensor.dims.clone(),
        values,
    })
}

fn load_f32(
    store: &OnnxInitializerStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<DenseTensor> {
    let values = store
        .required_f32(name, expected_shape)
        .map_err(map_onnx_error)?;
    validate_storage_len(name, expected_shape, values.len())?;
    Ok(DenseTensor {
        name: name.to_string(),
        shape: expected_shape.to_vec(),
        values,
    })
}

fn validate_storage_len(name: &str, shape: &[usize], actual_len: usize) -> Result<()> {
    let expected_len = shape.iter().product::<usize>();
    if actual_len != expected_len {
        return Err(VitsError::InvalidInput(format!(
            "tensor {name} storage has {actual_len} value(s), expected {expected_len} for shape {shape:?}"
        )));
    }
    Ok(())
}

fn expect_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(wrong_shape(name, actual, expected));
    }
    Ok(())
}

fn wrong_shape(name: &str, actual: &[usize], expected: &[usize]) -> VitsError {
    VitsError::InvalidInput(format!(
        "tensor {name} shape {actual:?} does not match expected {expected:?}"
    ))
}

fn sequence_indices(store: &OnnxInitializerStore, prefix: &str, suffix: &str) -> Vec<usize> {
    let mut indices = store
        .iter()
        .filter_map(|tensor| {
            let rest = tensor.name.strip_prefix(prefix)?;
            let index = rest.strip_suffix(suffix)?;
            index.parse::<usize>().ok()
        })
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn map_onnx_error(err: crate::models::onnx::OnnxLoadError) -> VitsError {
    VitsError::InvalidInput(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::models::onnx::{OnnxInitializerStore, OnnxTensor, OnnxTensorType};

    use super::*;

    #[test]
    fn reports_missing_tensor_in_handcrafted_store() {
        let store = OnnxInitializerStore::from_tensors([f32_tensor("sid", &[4, 2])]);
        let err = VitsWeights::from_initializers(
            &store,
            VitsWeightConfig {
                num_symbols: 4,
                num_speakers: 1,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("enc_p.proj.weight"));
    }

    #[test]
    fn reports_wrong_dtype_in_handcrafted_store() {
        let store = OnnxInitializerStore::from_tensors([OnnxTensor {
            name: "sid".to_string(),
            dims: vec![4, 2],
            data_type: OnnxTensorType::Int64,
            raw_data: Vec::new(),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: vec![0; 8],
        }]);
        let err = VitsWeights::from_initializers(
            &store,
            VitsWeightConfig {
                num_symbols: 4,
                num_speakers: 1,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("expected FLOAT"));
    }

    #[test]
    fn reports_wrong_shape_in_handcrafted_store() {
        let store = OnnxInitializerStore::from_tensors([f32_tensor("sid", &[3, 2])]);
        let err = VitsWeights::from_initializers(
            &store,
            VitsWeightConfig {
                num_symbols: 4,
                num_speakers: 1,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("shape [3, 2]"));
    }

    #[test]
    fn loads_deterministic_duration_predictor_from_handcrafted_store() {
        let store = OnnxInitializerStore::from_tensors([
            f32_tensor("dp.conv_1.weight", &[2, 3, 3]),
            f32_tensor("dp.conv_1.bias", &[2]),
            f32_tensor("dp.norm_1.gamma", &[2]),
            f32_tensor("dp.norm_1.beta", &[2]),
            f32_tensor("dp.conv_2.weight", &[2, 2, 3]),
            f32_tensor("dp.conv_2.bias", &[2]),
            f32_tensor("dp.norm_2.gamma", &[2]),
            f32_tensor("dp.norm_2.beta", &[2]),
            f32_tensor("dp.proj.weight", &[1, 2, 1]),
            f32_tensor("dp.proj.bias", &[1]),
        ]);

        let predictor = load_optional_deterministic_duration_predictor(&store, 3)
            .unwrap()
            .unwrap();

        assert_eq!(predictor.channels, 3);
        assert_eq!(predictor.filter_channels, 2);
        assert_eq!(predictor.kernel_size, 3);
    }

    #[test]
    fn generator_forward_runs_small_network() {
        let generator = GeneratorWeights {
            conv_pre: conv("pre.weight", &[1, 1, 1], vec![1.0], None),
            upsample_layers: vec![conv("up.weight", &[1, 1, 2], vec![1.0, 1.0], None)],
            resblocks: vec![GeneratorResBlockWeights {
                index: 0,
                convs: vec![
                    conv("rb.0.weight", &[1, 1, 1], vec![0.0], None),
                    conv("rb.1.weight", &[1, 1, 1], vec![0.0], None),
                ],
            }],
            conv_post: conv("post.weight", &[1, 1, 1], vec![1.0], None),
        };

        let audio = generator.infer(&[0.5], 1).unwrap();

        assert_eq!(audio.len(), 2);
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!((audio[0] - 0.5f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn text_encoder_forward_projects_embedding_stats() {
        let encoder = TextEncoderWeights {
            embedding: DenseTensor {
                name: "sid".to_string(),
                shape: vec![3, 1],
                values: vec![0.0, 2.0, 4.0],
            },
            projection: conv("enc_p.proj.weight", &[2, 1, 1], vec![1.0, -1.0], None),
            attention_layers: Vec::new(),
            ffn_layers: Vec::new(),
            norm_layers_1: Vec::new(),
            norm_layers_2: Vec::new(),
        };

        let out = encoder.infer(&[1, 2]).unwrap();

        assert_eq!(out.channels, 1);
        assert_eq!(out.frames, 2);
        assert_eq!(out.hidden, vec![2.0, 4.0]);
        assert_eq!(out.prior_mean, vec![2.0, 4.0]);
        assert_eq!(out.prior_log_scale, vec![-2.0, -4.0]);
    }

    #[test]
    fn expands_text_encoder_prior_by_durations() {
        let encoded = TextEncoderOutput {
            hidden: Vec::new(),
            prior_mean: vec![1.0, 2.0, 10.0, 20.0],
            prior_log_scale: vec![3.0, 4.0, 30.0, 40.0],
            channels: 2,
            frames: 2,
        };

        let expanded = encoded.expand_prior_by_durations(&[2, 1]).unwrap();

        assert_eq!(expanded.frames, 3);
        assert_eq!(expanded.mean, vec![1.0, 1.0, 2.0, 10.0, 10.0, 20.0]);
        assert_eq!(expanded.log_scale, vec![3.0, 3.0, 4.0, 30.0, 30.0, 40.0]);
    }

    #[test]
    fn residual_coupling_flow_reverse_runs_small_block() {
        let flow = ResidualCouplingFlowWeights {
            blocks: vec![ResidualCouplingBlockWeights {
                index: 0,
                pre: conv("flow.pre.weight", &[1, 1, 1], vec![1.0], None),
                post: conv("flow.post.weight", &[1, 1, 1], vec![0.0], None),
                wavenet: WavenetWeights {
                    in_layers: vec![conv("wn.in.weight", &[2, 1, 1], vec![0.0, 0.0], None)],
                    res_skip_layers: vec![conv("wn.skip.weight", &[1, 1, 1], vec![0.0], None)],
                },
            }],
        };

        let out = flow.reverse(&[1.0, 2.0, 10.0, 20.0], 2, 2).unwrap();

        assert_eq!(out, vec![10.0, 20.0, 1.0, 2.0]);
    }

    #[test]
    fn stochastic_duration_reverse_runs_small_predictor() {
        let predictor = StochasticDurationPredictorWeights {
            pre: conv("dp.pre.weight", &[1, 1, 1], vec![1.0], None),
            proj: conv("dp.proj.weight", &[1, 1, 1], vec![1.0], None),
            dds_convs: DdsConvWeights {
                layers: vec![DdsConvLayerWeights {
                    conv_sep: conv("dp.sep.weight", &[1, 1, 1], vec![0.0], Some(vec![0.0])),
                    conv_1x1: conv("dp.1x1.weight", &[1, 1, 1], vec![0.0], Some(vec![0.0])),
                    norm_1: norm("dp.norm1", 1),
                    norm_2: norm("dp.norm2", 1),
                }],
            },
            flows: Vec::new(),
            flow0_m: DenseTensor {
                name: "dp.flows.0.m".to_string(),
                shape: vec![2, 1],
                values: vec![0.0, 0.0],
            },
            flow0_scale: DenseTensor {
                name: "/dp/flows.0/Exp_output_0".to_string(),
                shape: vec![2, 1],
                values: vec![1.0, 1.0],
            },
        };
        let mut rng = DeterministicRng::new(7);

        let log_durations = predictor
            .reverse_log_durations(&[0.5, -0.25], 2, 0.1, &mut rng)
            .unwrap();

        assert_eq!(log_durations.len(), 2);
        assert!(log_durations.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn loads_local_piper_weight_smoke_if_model_exists() {
        let path = Path::new("models/piper/model.onnx");
        if !path.exists() {
            return;
        }

        let weights = VitsWeights::from_onnx_file(
            path,
            VitsWeightConfig {
                num_symbols: 256,
                num_speakers: 1,
            },
        )
        .unwrap();

        assert_eq!(weights.hidden_channels, 192);
        assert_eq!(weights.text_encoder.attention_layers.len(), 6);
        assert_eq!(weights.duration_predictor.flows.len(), 3);
        assert!(weights.deterministic_duration_predictor.is_none());
        assert_eq!(weights.residual_coupling_flow.blocks.len(), 4);
        assert_eq!(weights.generator.upsample_layers.len(), 3);
        assert!(weights.speaker_embedding.is_none());

        let encoded = weights.text_encoder.infer(&[1, 2, 3]).unwrap();
        assert_eq!(encoded.channels, 192);
        assert_eq!(encoded.frames, 3);
        assert_eq!(encoded.prior_mean.len(), 192 * 3);
        assert!(encoded.prior_mean.iter().all(|value| value.is_finite()));
        assert!(encoded
            .prior_log_scale
            .iter()
            .all(|value| value.is_finite()));

        let mut rng = DeterministicRng::new(1234);
        let log_durations = weights
            .duration_predictor
            .reverse_log_durations(&encoded.hidden, encoded.frames, 0.8, &mut rng)
            .unwrap();
        assert_eq!(log_durations.len(), encoded.frames);
        assert!(log_durations.iter().all(|value| value.is_finite()));

        let flowed = weights
            .residual_coupling_flow
            .reverse(&vec![0.0; 192 * 2], 192, 2)
            .unwrap();
        assert_eq!(flowed.len(), 192 * 2);
        assert!(flowed.iter().all(|value| value.is_finite()));
    }

    fn f32_tensor(name: &str, shape: &[usize]) -> OnnxTensor {
        OnnxTensor {
            name: name.to_string(),
            dims: shape.to_vec(),
            data_type: OnnxTensorType::Float32,
            raw_data: Vec::new(),
            float_data: vec![0.0; shape.iter().product()],
            int32_data: Vec::new(),
            int64_data: Vec::new(),
        }
    }

    fn conv(name: &str, shape: &[usize], values: Vec<f32>, bias: Option<Vec<f32>>) -> ConvWeights {
        ConvWeights {
            weight: DenseTensor {
                name: name.to_string(),
                shape: shape.to_vec(),
                values,
            },
            bias: bias.map(|values| DenseTensor {
                name: format!("{name}.bias"),
                shape: vec![values.len()],
                values,
            }),
        }
    }

    fn norm(name: &str, channels: usize) -> LayerNormWeights {
        LayerNormWeights {
            gamma: DenseTensor {
                name: format!("{name}.gamma"),
                shape: vec![channels],
                values: vec![1.0; channels],
            },
            beta: DenseTensor {
                name: format!("{name}.beta"),
                shape: vec![channels],
                values: vec![0.0; channels],
            },
        }
    }
}
