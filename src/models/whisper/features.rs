use std::f32::consts::PI;

use super::preprocessor::WhisperPreprocessorConfig;
use super::{Result, WhisperError};

#[derive(Clone, Debug, PartialEq)]
pub struct LogMelSpectrogram {
    pub values: Vec<f32>,
    pub n_mels: usize,
    pub n_frames: usize,
}

impl LogMelSpectrogram {
    pub fn row(&self, mel: usize) -> &[f32] {
        let start = mel * self.n_frames;
        &self.values[start..start + self.n_frames]
    }
}

pub fn pad_or_trim(samples: &[f32], n_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_samples];
    let len = samples.len().min(n_samples);
    out[..len].copy_from_slice(&samples[..len]);
    out
}

pub fn log_mel_spectrogram(
    samples: &[f32],
    config: &WhisperPreprocessorConfig,
) -> Result<LogMelSpectrogram> {
    config.validate()?;
    let audio = pad_or_trim(samples, config.n_samples);
    let power = stft_power_spectrogram(&audio, config.n_fft, config.hop_length);
    let filters = mel_filterbank(config.sample_rate, config.n_fft, config.n_mels);
    let freq_bins = config.n_fft / 2 + 1;
    let stft_frames = power.len() / freq_bins;
    if stft_frames < config.n_frames {
        return Err(WhisperError::Audio(format!(
            "STFT produced {stft_frames} frames, need {}",
            config.n_frames
        )));
    }

    let mut values = vec![0.0f32; config.n_mels * config.n_frames];
    for mel in 0..config.n_mels {
        for frame in 0..config.n_frames {
            let mut energy = 0.0f32;
            for bin in 0..freq_bins {
                energy += filters[mel * freq_bins + bin] * power[frame * freq_bins + bin];
            }
            values[mel * config.n_frames + frame] = energy.max(1e-10).log10();
        }
    }

    let max_log = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let floor = max_log - 8.0;
    for value in &mut values {
        *value = ((*value).max(floor) + 4.0) / 4.0;
    }

    Ok(LogMelSpectrogram {
        values,
        n_mels: config.n_mels,
        n_frames: config.n_frames,
    })
}

fn stft_power_spectrogram(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; samples.len() + 2 * pad];
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    let frames = (padded.len() - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let window = hann_window(n_fft);
    let twiddle = dft_twiddle(freq_bins, n_fft);
    let mut out = vec![0.0f32; frames * freq_bins];

    for frame in 0..frames {
        let start = frame * hop_length;
        let frame_samples = &padded[start..start + n_fft];
        for bin in 0..freq_bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for n in 0..n_fft {
                let sample = frame_samples[n] * window[n];
                let (cos, sin) = twiddle[bin * n_fft + n];
                re += sample * cos;
                im -= sample * sin;
            }
            out[frame * freq_bins + bin] = re * re + im * im;
        }
    }
    out
}

fn hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|i| (PI * i as f32 / n_fft as f32).sin().powi(2))
        .collect()
}

fn dft_twiddle(freq_bins: usize, n_fft: usize) -> Vec<(f32, f32)> {
    let mut values = Vec::with_capacity(freq_bins * n_fft);
    for bin in 0..freq_bins {
        for n in 0..n_fft {
            let angle = 2.0 * PI * bin as f32 * n as f32 / n_fft as f32;
            values.push((angle.cos(), angle.sin()));
        }
    }
    values
}

fn mel_filterbank(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let freq_bins = n_fft / 2 + 1;
    let f_min = 0.0f32;
    let f_max = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_to_hz(mel)
        })
        .collect::<Vec<_>>();
    let fft_freqs = (0..freq_bins)
        .map(|i| sample_rate as f32 * i as f32 / n_fft as f32)
        .collect::<Vec<_>>();

    let mut filters = vec![0.0f32; n_mels * freq_bins];
    for mel in 0..n_mels {
        let lower = mel_points[mel];
        let center = mel_points[mel + 1];
        let upper = mel_points[mel + 2];
        for (bin, freq) in fft_freqs.iter().copied().enumerate() {
            let lower_slope = (freq - lower) / (center - lower);
            let upper_slope = (upper - freq) / (upper - center);
            filters[mel * freq_bins + bin] = lower_slope.min(upper_slope).max(0.0);
        }
        let enorm = 2.0 / (upper - lower);
        for bin in 0..freq_bins {
            filters[mel * freq_bins + bin] *= enorm;
        }
    }
    filters
}

fn hz_to_mel(freq: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    if freq < min_log_hz {
        freq / f_sp
    } else {
        let logstep = 6.4f32.ln() / 27.0;
        min_log_mel + (freq / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        let logstep = 6.4f32.ln() / 27.0;
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::whisper::load_wav_pcm;

    #[test]
    fn extracts_jfk_log_mel_features_with_whisper_shape() -> Result<()> {
        let audio = load_wav_pcm("tests/data/audio/jfk_16khz_mono.wav")?;
        let config = WhisperPreprocessorConfig::default();

        let features = log_mel_spectrogram(&audio.samples, &config)?;

        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 3000);
        assert_eq!(features.values.len(), 80 * 3000);
        assert!(features.values.iter().all(|value| value.is_finite()));
        assert!((features.values[0] - -0.538_741_95).abs() < 1e-5);
        assert!((features.values[40 * 3000 + 100] - 0.145_014_29).abs() < 1e-5);
        assert!((features.values[79 * 3000 + 2999] - -0.538_741_95).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn jfk_log_mel_matches_openai_reference_snapshot() -> Result<()> {
        let audio = load_wav_pcm("tests/data/audio/jfk_16khz_mono.wav")?;
        let config = WhisperPreprocessorConfig::default();
        let features = log_mel_spectrogram(&audio.samples, &config)?;
        let reference = std::fs::read("tests/data/whisper/jfk_logmel_openai_ref_f32.bin")
            .map_err(|err| WhisperError::Asset(err.to_string()))?;
        assert_eq!(reference.len(), features.values.len() * 4);

        let mut max_abs_diff = 0.0f32;
        for (i, actual) in features.values.iter().copied().enumerate() {
            let start = i * 4;
            let expected = f32::from_le_bytes([
                reference[start],
                reference[start + 1],
                reference[start + 2],
                reference[start + 3],
            ]);
            max_abs_diff = max_abs_diff.max((actual - expected).abs());
        }

        assert!(
            max_abs_diff <= 2e-3,
            "max log-mel abs diff {max_abs_diff} exceeded tolerance"
        );
        Ok(())
    }

    #[test]
    fn extracts_micro_machines_log_mel_features_with_whisper_shape() -> Result<()> {
        let audio = load_wav_pcm("tests/data/audio/micro_machines_16khz_mono.wav")?;
        let config = WhisperPreprocessorConfig::default();

        let features = log_mel_spectrogram(&audio.samples, &config)?;

        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 3000);
        assert_eq!(features.values.len(), 80 * 3000);
        assert!(features.values.iter().all(|value| value.is_finite()));
        assert!((features.values[0] - -0.384_043_46).abs() < 1e-5);
        assert!((features.values[40 * 3000 + 100] - 0.632_712_36).abs() < 1e-5);
        assert!((features.values[79 * 3000 + 2999] - -0.761_784_55).abs() < 1e-5);
        Ok(())
    }
}
