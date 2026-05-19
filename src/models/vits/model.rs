use std::f32::consts::PI;

use super::{
    channel_layer_norm_in_place, conv1d, leaky_relu_in_place, Conv1dParams, Result, VitsError,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VitsSynthesisScales {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

impl Default for VitsSynthesisScales {
    fn default() -> Self {
        Self {
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_w: 0.8,
        }
    }
}

pub fn infer_frame_count(log_durations: &[f32], length_scale: f32) -> usize {
    log_durations
        .iter()
        .map(|value| (value.exp() * length_scale).ceil().max(0.0) as usize)
        .sum::<usize>()
        .max(1)
}

pub fn log_durations_to_durations(log_durations: &[f32], length_scale: f32) -> Vec<usize> {
    log_durations
        .iter()
        .map(|value| (value.exp() * length_scale).ceil().max(0.0) as usize)
        .collect()
}

pub fn expand_by_durations(
    values: &[f32],
    channels: usize,
    durations: &[usize],
) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(VitsError::InvalidInput("channels must be > 0".to_string()));
    }
    if values.len() != channels * durations.len() {
        return Err(VitsError::InvalidInput(format!(
            "values length {} does not match {} channels x {} durations",
            values.len(),
            channels,
            durations.len()
        )));
    }

    let frames = durations.iter().sum::<usize>();
    let mut out = vec![0.0f32; channels * frames];
    let mut dst_t = 0;
    for (src_t, duration) in durations.iter().copied().enumerate() {
        for _ in 0..duration {
            for c in 0..channels {
                out[c * frames + dst_t] = values[c * durations.len() + src_t];
            }
            dst_t += 1;
        }
    }
    Ok(out)
}

pub fn duration_path(durations: &[usize]) -> Vec<Vec<bool>> {
    let frames = durations.iter().sum();
    let mut path = vec![vec![false; durations.len()]; frames];
    let mut frame = 0;
    for (token, duration) in durations.iter().copied().enumerate() {
        for _ in 0..duration {
            path[frame][token] = true;
            frame += 1;
        }
    }
    path
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeterministicDurationPredictorWeights {
    pub channels: usize,
    pub filter_channels: usize,
    pub kernel_size: usize,
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta: Vec<f32>,
    pub proj_weight: Vec<f32>,
    pub proj_bias: f32,
}

impl DeterministicDurationPredictorWeights {
    pub fn infer(&self, input: &[f32], frames: usize) -> Result<Vec<f32>> {
        if input.len() != self.channels * frames {
            return Err(VitsError::InvalidInput(format!(
                "duration predictor input length {} does not match {} channels x {} frames",
                input.len(),
                self.channels,
                frames
            )));
        }

        let conv1_params = Conv1dParams {
            in_channels: self.channels,
            out_channels: self.filter_channels,
            kernel_size: self.kernel_size,
            stride: 1,
            padding: self.kernel_size / 2,
            dilation: 1,
            groups: 1,
        };
        let mut hidden = conv1d(
            input,
            frames,
            conv1_params,
            &self.conv1_weight,
            Some(&self.conv1_bias),
        )?;
        leaky_relu_in_place(&mut hidden, 0.0);
        channel_layer_norm_in_place(
            &mut hidden,
            self.filter_channels,
            frames,
            &self.norm1_gamma,
            &self.norm1_beta,
            1e-5,
        )?;

        let conv2_params = Conv1dParams {
            in_channels: self.filter_channels,
            out_channels: self.filter_channels,
            kernel_size: self.kernel_size,
            stride: 1,
            padding: self.kernel_size / 2,
            dilation: 1,
            groups: 1,
        };
        hidden = conv1d(
            &hidden,
            frames,
            conv2_params,
            &self.conv2_weight,
            Some(&self.conv2_bias),
        )?;
        leaky_relu_in_place(&mut hidden, 0.0);
        channel_layer_norm_in_place(
            &mut hidden,
            self.filter_channels,
            frames,
            &self.norm2_gamma,
            &self.norm2_beta,
            1e-5,
        )?;

        let proj_params = Conv1dParams {
            in_channels: self.filter_channels,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        conv1d(
            &hidden,
            frames,
            proj_params,
            &self.proj_weight,
            Some(&[self.proj_bias]),
        )
    }
}

pub fn debug_synthesize_phoneme_ids(
    phoneme_ids: &[usize],
    sample_rate: usize,
    scales: VitsSynthesisScales,
    speaker_id: Option<usize>,
) -> Result<Vec<f32>> {
    if sample_rate == 0 {
        return Err(VitsError::InvalidInput(
            "sample_rate must be > 0".to_string(),
        ));
    }
    if phoneme_ids.is_empty() {
        return Err(VitsError::InvalidInput(
            "phoneme id input must not be empty".to_string(),
        ));
    }

    let samples_per_symbol =
        ((0.055 * scales.length_scale.max(0.05)) * sample_rate as f32).ceil() as usize;
    let silence_samples = (0.008 * sample_rate as f32).ceil() as usize;
    let speaker_offset = speaker_id.unwrap_or(0) as f32 * 7.0;
    let mut audio = Vec::with_capacity(phoneme_ids.len() * (samples_per_symbol + silence_samples));

    for id in phoneme_ids {
        let freq = 180.0 + ((*id % 48) as f32 * 11.0) + speaker_offset;
        for i in 0..samples_per_symbol {
            let t = i as f32 / sample_rate as f32;
            let envelope = raised_cosine_envelope(i, samples_per_symbol);
            audio.push((2.0 * PI * freq * t).sin() * 0.16 * envelope);
        }
        audio.extend(std::iter::repeat(0.0).take(silence_samples));
    }
    Ok(audio)
}

fn raised_cosine_envelope(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    let phase = index as f32 / (len - 1) as f32;
    (PI * phase).sin().max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_vits_duration_rule() {
        let durations = [0.0f32, 1.0f32.ln(), 2.1f32.ln()];

        assert_eq!(infer_frame_count(&durations, 1.0), 5);
        assert_eq!(log_durations_to_durations(&durations, 1.0), vec![1, 1, 3]);
    }

    #[test]
    fn expands_channel_major_values_by_duration() {
        let values = vec![
            1.0, 2.0, 3.0, //
            10.0, 20.0, 30.0,
        ];
        let out = expand_by_durations(&values, 2, &[2, 0, 1]).unwrap();

        assert_eq!(out, vec![1.0, 1.0, 3.0, 10.0, 10.0, 30.0]);
    }

    #[test]
    fn builds_monotonic_duration_path() {
        let path = duration_path(&[2, 1]);

        assert_eq!(
            path,
            vec![vec![true, false], vec![true, false], vec![false, true]]
        );
    }

    #[test]
    fn deterministic_duration_predictor_runs_small_network() {
        let weights = DeterministicDurationPredictorWeights {
            channels: 1,
            filter_channels: 1,
            kernel_size: 1,
            conv1_weight: vec![1.0],
            conv1_bias: vec![0.0],
            norm1_gamma: vec![1.0],
            norm1_beta: vec![0.0],
            conv2_weight: vec![1.0],
            conv2_bias: vec![0.0],
            norm2_gamma: vec![1.0],
            norm2_beta: vec![0.0],
            proj_weight: vec![1.0],
            proj_bias: 0.25,
        };

        let out = weights.infer(&[1.0, 2.0], 2).unwrap();

        assert_eq!(out, vec![0.25, 0.25]);
    }

    #[test]
    fn debug_synthesis_returns_finite_audio() {
        let audio =
            debug_synthesize_phoneme_ids(&[1, 2], 1_000, VitsSynthesisScales::default(), None)
                .unwrap();

        assert!(!audio.is_empty());
        assert!(audio.iter().all(|sample| sample.is_finite()));
    }
}
