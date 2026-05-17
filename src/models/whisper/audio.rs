use std::fs;
use std::path::{Path, PathBuf};

use super::{Result, WhisperError};

#[derive(Clone, Debug, PartialEq)]
pub struct PcmAudio {
    pub path: PathBuf,
    pub sample_rate: usize,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl PcmAudio {
    pub fn duration_seconds(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

pub fn load_wav_pcm(path: impl AsRef<Path>) -> Result<PcmAudio> {
    let path = path.as_ref();
    #[cfg(feature = "audio-formats")]
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("flac"))
    {
        return load_flac_pcm(path);
    }

    let data = fs::read(path).map_err(|err| {
        WhisperError::Audio(format!(
            "failed to read audio file {}: {err}",
            path.display()
        ))
    })?;
    parse_wav_pcm(path, &data)
}

pub fn load_wav_pcm_bytes(display_path: impl Into<PathBuf>, data: &[u8]) -> Result<PcmAudio> {
    let path = display_path.into();
    parse_wav_pcm(&path, data)
}

#[cfg(feature = "audio-formats")]
fn load_flac_pcm(path: &Path) -> Result<PcmAudio> {
    let mut reader = claxon::FlacReader::open(path).map_err(|err| {
        WhisperError::Audio(format!(
            "failed to decode FLAC file {}: {err}",
            path.display()
        ))
    })?;
    let info = reader.streaminfo();
    let channels = info.channels as usize;
    if channels == 0 {
        return Err(WhisperError::Audio(format!(
            "{} has zero channels",
            path.display()
        )));
    }
    let bits_per_sample = info.bits_per_sample;
    if bits_per_sample == 0 || bits_per_sample > 32 {
        return Err(WhisperError::Audio(format!(
            "{} has unsupported FLAC bit depth {}",
            path.display(),
            bits_per_sample
        )));
    }
    let scale = (1_i64 << (bits_per_sample - 1)) as f32;
    let mut samples = Vec::new();
    let mut frame = Vec::with_capacity(channels);
    for sample in reader.samples() {
        let sample = sample.map_err(|err| {
            WhisperError::Audio(format!(
                "failed to decode FLAC sample {}: {err}",
                path.display()
            ))
        })?;
        frame.push(sample as f32 / scale);
        if frame.len() == channels {
            samples.push(frame.iter().sum::<f32>() / channels as f32);
            frame.clear();
        }
    }
    if !frame.is_empty() {
        return Err(WhisperError::Audio(format!(
            "{} ended with a partial FLAC frame",
            path.display()
        )));
    }

    Ok(PcmAudio {
        path: path.to_path_buf(),
        sample_rate: info.sample_rate as usize,
        channels,
        samples,
    })
}

fn parse_wav_pcm(path: &Path, data: &[u8]) -> Result<PcmAudio> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(WhisperError::Audio(format!(
            "{} is not a RIFF/WAVE PCM file",
            path.display()
        )));
    }

    let mut offset = 12;
    let mut format = None;
    let mut pcm_data = None;
    while offset + 8 <= data.len() {
        let id = &data[offset..offset + 4];
        let len = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        offset += 8;
        if offset + len > data.len() {
            return Err(WhisperError::Audio(format!(
                "{} has a truncated WAV chunk",
                path.display()
            )));
        }
        match id {
            b"fmt " => format = Some(parse_fmt_chunk(path, &data[offset..offset + len])?),
            b"data" => pcm_data = Some(&data[offset..offset + len]),
            _ => {}
        }
        offset += len + (len % 2);
    }

    let format = format.ok_or_else(|| {
        WhisperError::Audio(format!("{} is missing a WAV fmt chunk", path.display()))
    })?;
    let pcm_data = pcm_data.ok_or_else(|| {
        WhisperError::Audio(format!("{} is missing a WAV data chunk", path.display()))
    })?;
    if format.audio_format != 1 {
        return Err(WhisperError::Audio(format!(
            "{} uses WAV format {}, only PCM is supported",
            path.display(),
            format.audio_format
        )));
    }
    if format.bits_per_sample != 16 {
        return Err(WhisperError::Audio(format!(
            "{} uses {} bits per sample, only 16-bit PCM is supported",
            path.display(),
            format.bits_per_sample
        )));
    }
    if format.channels == 0 {
        return Err(WhisperError::Audio(format!(
            "{} has zero channels",
            path.display()
        )));
    }

    let channels = format.channels as usize;
    let bytes_per_frame = channels * 2;
    if pcm_data.len() % bytes_per_frame != 0 {
        return Err(WhisperError::Audio(format!(
            "{} has a data chunk length that is not aligned to {} channels",
            path.display(),
            channels
        )));
    }

    let mut samples = Vec::with_capacity(pcm_data.len() / bytes_per_frame);
    for frame in pcm_data.chunks_exact(bytes_per_frame) {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            let i = ch * 2;
            let sample = i16::from_le_bytes([frame[i], frame[i + 1]]) as f32 / 32768.0;
            sum += sample;
        }
        samples.push(sum / channels as f32);
    }

    Ok(PcmAudio {
        path: path.to_path_buf(),
        sample_rate: format.sample_rate as usize,
        channels,
        samples,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn parse_fmt_chunk(path: &Path, data: &[u8]) -> Result<WavFormat> {
    if data.len() < 16 {
        return Err(WhisperError::Audio(format!(
            "{} has a truncated WAV fmt chunk",
            path.display()
        )));
    }
    Ok(WavFormat {
        audio_format: u16::from_le_bytes([data[0], data[1]]),
        channels: u16::from_le_bytes([data[2], data[3]]),
        sample_rate: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
        bits_per_sample: u16::from_le_bytes([data[14], data[15]]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_jfk_fixture_as_16khz_mono_pcm() -> Result<()> {
        let audio = load_wav_pcm("tests/data/audio/jfk_16khz_mono.wav")?;

        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 176_000);
        assert!((audio.duration_seconds() - 11.0).abs() < 0.001);
        Ok(())
    }

    #[test]
    fn loads_micro_machines_fixture_as_near_full_window() -> Result<()> {
        let audio = load_wav_pcm("tests/data/audio/micro_machines_16khz_mono.wav")?;

        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert!(audio.samples.len() > 478_000);
        assert!(audio.samples.len() < 479_000);
        Ok(())
    }

    #[test]
    fn downmixes_stereo_wav() -> Result<()> {
        let wav = tiny_stereo_wav();
        let audio = load_wav_pcm_bytes("memory.wav", &wav)?;

        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples, vec![0.25, 0.5]);
        Ok(())
    }

    #[cfg(feature = "audio-formats")]
    #[test]
    fn optional_feature_reports_missing_flac_fixture() {
        let err = load_wav_pcm("tests/data/audio/jfk.flac").unwrap_err();
        assert!(err.to_string().contains("failed to decode FLAC file"));
    }

    fn tiny_stereo_wav() -> Vec<u8> {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&44u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&64_000u32.to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&8u32.to_le_bytes());
        wav.extend_from_slice(&0i16.to_le_bytes());
        wav.extend_from_slice(&16_384i16.to_le_bytes());
        wav.extend_from_slice(&16_384i16.to_le_bytes());
        wav.extend_from_slice(&16_384i16.to_le_bytes());
        wav
    }
}
