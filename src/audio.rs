use std::error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

#[derive(Debug)]
pub enum AudioError {
    Device(String),
    Io(String),
    Wav(String),
    Unsupported(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::Device(msg) => write!(f, "audio device error: {msg}"),
            AudioError::Io(msg) => write!(f, "audio I/O error: {msg}"),
            AudioError::Wav(msg) => write!(f, "WAV error: {msg}"),
            AudioError::Unsupported(msg) => write!(f, "unsupported audio operation: {msg}"),
        }
    }
}

impl error::Error for AudioError {}

pub type AudioResult<T> = std::result::Result<T, AudioError>;

#[derive(Clone, Debug, PartialEq)]
pub struct PcmAudio {
    pub path: PathBuf,
    pub sample_rate: usize,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl PcmAudio {
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputDeviceInfo {
    pub index: usize,
    pub name: String,
    pub is_default: bool,
}

pub fn list_input_devices() -> AudioResult<Vec<InputDeviceInfo>> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let devices = host
        .input_devices()
        .map_err(|err| AudioError::Device(format!("failed to enumerate input devices: {err}")))?;
    let mut infos = Vec::new();
    for (index, device) in devices.enumerate() {
        let name = device
            .name()
            .unwrap_or_else(|err| format!("<unavailable name: {err}>"));
        let is_default = default_name
            .as_ref()
            .is_some_and(|default| default == &name);
        infos.push(InputDeviceInfo {
            index,
            name,
            is_default,
        });
    }
    Ok(infos)
}

pub fn record_input_device(
    device_index: Option<usize>,
    duration: Duration,
) -> AudioResult<PcmAudio> {
    if duration.is_zero() {
        return Err(AudioError::Device(
            "recording duration must be greater than zero".to_string(),
        ));
    }

    let host = cpal::default_host();
    let device = match device_index {
        Some(index) => host
            .input_devices()
            .map_err(|err| AudioError::Device(format!("failed to enumerate input devices: {err}")))?
            .nth(index)
            .ok_or_else(|| AudioError::Device(format!("input device index {index} not found")))?,
        None => host.default_input_device().ok_or_else(|| {
            AudioError::Device("no default input device is available".to_string())
        })?,
    };

    let config = device
        .default_input_config()
        .map_err(|err| AudioError::Device(format!("failed to read default input config: {err}")))?;
    let sample_rate = config.sample_rate().0 as usize;
    let channels = config.channels() as usize;
    if channels == 0 {
        return Err(AudioError::Device(
            "selected input device reports zero channels".to_string(),
        ));
    }

    let wanted_samples = ((duration.as_secs_f64() * sample_rate as f64).ceil() as usize).max(1);
    let (sender, receiver) = mpsc::sync_channel::<Vec<f32>>(64);
    let overflowed = Arc::new(AtomicBool::new(false));
    let stream_config = config.config();
    let err_fn = |err| eprintln!("audio input stream error: {err}");

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build_input_stream::<f32, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_f32,
        )?,
        cpal::SampleFormat::F64 => build_input_stream::<f64, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_f64,
        )?,
        cpal::SampleFormat::I8 => build_input_stream::<i8, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_i8,
        )?,
        cpal::SampleFormat::I16 => build_input_stream::<i16, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_i16,
        )?,
        cpal::SampleFormat::I32 => build_input_stream::<i32, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_i32,
        )?,
        cpal::SampleFormat::I64 => build_input_stream::<i64, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_i64,
        )?,
        cpal::SampleFormat::U8 => build_input_stream::<u8, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_u8,
        )?,
        cpal::SampleFormat::U16 => build_input_stream::<u16, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_u16,
        )?,
        cpal::SampleFormat::U32 => build_input_stream::<u32, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_u32,
        )?,
        cpal::SampleFormat::U64 => build_input_stream::<u64, _, _>(
            &device,
            &stream_config,
            channels,
            sender,
            overflowed.clone(),
            err_fn,
            convert_u64,
        )?,
        sample_format => {
            return Err(AudioError::Unsupported(format!(
                "input sample format {sample_format:?} is not supported"
            )));
        }
    };

    stream
        .play()
        .map_err(|err| AudioError::Device(format!("failed to start input stream: {err}")))?;

    let deadline = Instant::now() + duration + Duration::from_secs(1);
    let mut samples = Vec::with_capacity(wanted_samples);
    while samples.len() < wanted_samples {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match receiver.recv_timeout(deadline - now) {
            Ok(mut chunk) => samples.append(&mut chunk),
            Err(mpsc::RecvTimeoutError::Timeout) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(AudioError::Device(
                    "audio input stream stopped while recording".to_string(),
                ));
            }
        }
    }
    drop(stream);

    if overflowed.load(Ordering::Relaxed) {
        return Err(AudioError::Device(
            "audio input buffer overflowed while recording".to_string(),
        ));
    }
    if samples.len() < wanted_samples {
        return Err(AudioError::Device(format!(
            "recorded {} samples, expected at least {wanted_samples}",
            samples.len()
        )));
    }
    samples.truncate(wanted_samples);

    Ok(PcmAudio {
        path: PathBuf::from("<microphone>"),
        sample_rate,
        channels,
        samples,
    })
}

fn build_input_stream<T, F, E>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    sender: mpsc::SyncSender<Vec<f32>>,
    overflowed: Arc<AtomicBool>,
    err_fn: E,
    convert: F,
) -> AudioResult<cpal::Stream>
where
    T: cpal::SizedSample,
    F: Fn(T) -> f32 + Send + Copy + 'static,
    E: FnMut(cpal::StreamError) + Send + 'static,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                let mut chunk = Vec::with_capacity(data.len() / channels);
                for frame in data.chunks_exact(channels) {
                    chunk.push(downmix_frame(frame.iter().copied().map(convert), channels));
                }
                if !chunk.is_empty() && sender.try_send(chunk).is_err() {
                    overflowed.store(true, Ordering::Relaxed);
                }
            },
            err_fn,
            None,
        )
        .map_err(|err| AudioError::Device(format!("failed to build input stream: {err}")))
}

pub fn load_wav_pcm(path: impl AsRef<Path>) -> AudioResult<PcmAudio> {
    let path = path.as_ref();
    let data = fs::read(path)
        .map_err(|err| AudioError::Io(format!("failed to read {}: {err}", path.display())))?;
    parse_wav_pcm(path, &data)
}

pub fn load_wav_pcm_bytes(display_path: impl Into<PathBuf>, data: &[u8]) -> AudioResult<PcmAudio> {
    let path = display_path.into();
    parse_wav_pcm(&path, data)
}

pub fn write_wav_pcm16(path: impl AsRef<Path>, audio: &PcmAudio) -> AudioResult<()> {
    let path = path.as_ref();
    let mut file = fs::File::create(path)
        .map_err(|err| AudioError::Io(format!("failed to create {}: {err}", path.display())))?;
    write_wav_pcm16_to(&mut file, audio)
        .map_err(|err| AudioError::Io(format!("failed to write {}: {err}", path.display())))
}

fn write_wav_pcm16_to(mut writer: impl Write, audio: &PcmAudio) -> io::Result<()> {
    let channels = 1u16;
    let sample_rate = u32::try_from(audio.sample_rate)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "sample rate is too large"))?;
    let sample_count = u32::try_from(audio.samples.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "audio is too long"))?;
    let bytes_per_sample = 2u16;
    let data_len = sample_count * u32::from(bytes_per_sample);
    let riff_len = 36u32
        .checked_add(data_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "WAV is too large"))?;

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_len.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(
        &(sample_rate * u32::from(channels) * u32::from(bytes_per_sample)).to_le_bytes(),
    )?;
    writer.write_all(&(channels * bytes_per_sample).to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    for sample in &audio.samples {
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_all(&scaled.to_le_bytes())?;
    }
    Ok(())
}

pub fn resample_linear(audio: &PcmAudio, target_sample_rate: usize) -> AudioResult<PcmAudio> {
    if target_sample_rate == 0 {
        return Err(AudioError::Unsupported(
            "target sample rate must be greater than zero".to_string(),
        ));
    }
    if audio.sample_rate == target_sample_rate {
        return Ok(audio.clone());
    }
    if audio.samples.is_empty() {
        return Ok(PcmAudio {
            path: audio.path.clone(),
            sample_rate: target_sample_rate,
            channels: 1,
            samples: Vec::new(),
        });
    }

    let output_len =
        ((audio.samples.len() as u128 * target_sample_rate as u128 + audio.sample_rate as u128 - 1)
            / audio.sample_rate as u128) as usize;
    let ratio = audio.sample_rate as f64 / target_sample_rate as f64;
    let mut samples = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src = i as f64 * ratio;
        let left = src.floor() as usize;
        let right = (left + 1).min(audio.samples.len() - 1);
        let frac = (src - left as f64) as f32;
        samples.push(audio.samples[left] * (1.0 - frac) + audio.samples[right] * frac);
    }

    Ok(PcmAudio {
        path: audio.path.clone(),
        sample_rate: target_sample_rate,
        channels: 1,
        samples,
    })
}

pub fn inspect_wav(path: impl AsRef<Path>) -> AudioResult<AudioInspection> {
    let audio = load_wav_pcm(path)?;
    Ok(AudioInspection {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        sample_count: audio.samples.len(),
        duration_seconds: audio.duration_seconds(),
        format: "PCM WAV 16-bit".to_string(),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudioInspection {
    pub sample_rate: usize,
    pub channels: usize,
    pub sample_count: usize,
    pub duration_seconds: f32,
    pub format: String,
}

fn parse_wav_pcm(path: &Path, data: &[u8]) -> AudioResult<PcmAudio> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(AudioError::Wav(format!(
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
            return Err(AudioError::Wav(format!(
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

    let format = format
        .ok_or_else(|| AudioError::Wav(format!("{} is missing a WAV fmt chunk", path.display())))?;
    let pcm_data = pcm_data.ok_or_else(|| {
        AudioError::Wav(format!("{} is missing a WAV data chunk", path.display()))
    })?;
    if format.audio_format != 1 {
        return Err(AudioError::Wav(format!(
            "{} uses WAV format {}, only PCM is supported",
            path.display(),
            format.audio_format
        )));
    }
    if format.bits_per_sample != 16 {
        return Err(AudioError::Wav(format!(
            "{} uses {} bits per sample, only 16-bit PCM is supported",
            path.display(),
            format.bits_per_sample
        )));
    }
    if format.channels == 0 {
        return Err(AudioError::Wav(format!(
            "{} has zero channels",
            path.display()
        )));
    }

    let channels = format.channels as usize;
    let bytes_per_frame = channels * 2;
    if pcm_data.len() % bytes_per_frame != 0 {
        return Err(AudioError::Wav(format!(
            "{} has a data chunk length that is not aligned to {} channels",
            path.display(),
            channels
        )));
    }

    let mut samples = Vec::with_capacity(pcm_data.len() / bytes_per_frame);
    for frame in pcm_data.chunks_exact(bytes_per_frame) {
        let values = (0..channels).map(|ch| {
            let i = ch * 2;
            convert_i16(i16::from_le_bytes([frame[i], frame[i + 1]]))
        });
        samples.push(downmix_frame(values, channels));
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

fn parse_fmt_chunk(path: &Path, data: &[u8]) -> AudioResult<WavFormat> {
    if data.len() < 16 {
        return Err(AudioError::Wav(format!(
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

pub fn downmix_interleaved(samples: &[f32], channels: usize) -> AudioResult<Vec<f32>> {
    if channels == 0 {
        return Err(AudioError::Unsupported(
            "channel count must be greater than zero".to_string(),
        ));
    }
    if samples.len() % channels != 0 {
        return Err(AudioError::Unsupported(format!(
            "{} samples are not divisible by {channels} channels",
            samples.len()
        )));
    }
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| downmix_frame(frame.iter().copied(), channels))
        .collect())
}

fn downmix_frame(samples: impl Iterator<Item = f32>, channels: usize) -> f32 {
    samples.sum::<f32>() / channels as f32
}

pub fn convert_i16(sample: i16) -> f32 {
    sample as f32 / 32768.0
}

pub fn convert_u16(sample: u16) -> f32 {
    sample as f32 / 32768.0 - 1.0
}

fn convert_f32(sample: f32) -> f32 {
    sample.clamp(-1.0, 1.0)
}

fn convert_f64(sample: f64) -> f32 {
    (sample as f32).clamp(-1.0, 1.0)
}

fn convert_i8(sample: i8) -> f32 {
    sample as f32 / 128.0
}

fn convert_i32(sample: i32) -> f32 {
    sample as f32 / 2_147_483_648.0
}

fn convert_i64(sample: i64) -> f32 {
    sample as f32 / 9_223_372_036_854_775_808.0
}

fn convert_u8(sample: u8) -> f32 {
    sample as f32 / 128.0 - 1.0
}

fn convert_u32(sample: u32) -> f32 {
    sample as f32 / 2_147_483_648.0 - 1.0
}

fn convert_u64(sample: u64) -> f32 {
    sample as f32 / 9_223_372_036_854_775_808.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmixes_interleaved_frames() {
        let mono = downmix_interleaved(&[1.0, -1.0, 0.25, 0.75], 2).unwrap();
        assert_eq!(mono, vec![0.0, 0.5]);
    }

    #[test]
    fn converts_integer_samples_to_normalized_f32() {
        assert_eq!(convert_i16(0), 0.0);
        assert!((convert_i16(i16::MAX) - 0.9999695).abs() < 0.000001);
        assert_eq!(convert_i16(i16::MIN), -1.0);
        assert_eq!(convert_u16(0), -1.0);
        assert!((convert_u16(u16::MAX) - 0.9999695).abs() < 0.000001);
    }

    #[test]
    fn resamples_linear_output_length_and_shape() {
        let audio = PcmAudio {
            path: PathBuf::from("memory.wav"),
            sample_rate: 4,
            channels: 1,
            samples: vec![0.0, 1.0, 0.0, -1.0],
        };
        let resampled = resample_linear(&audio, 8).unwrap();
        assert_eq!(resampled.sample_rate, 8);
        assert_eq!(resampled.channels, 1);
        assert_eq!(resampled.samples.len(), 8);
        assert!((resampled.samples[1] - 0.5).abs() < 0.000001);
        assert!((resampled.samples[2] - 1.0).abs() < 0.000001);
    }

    #[test]
    fn inspects_wav_fixture_without_microphone() {
        let info = inspect_wav("tests/data/audio/jfk_16khz_mono.wav").unwrap();
        assert_eq!(info.sample_rate, 16_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.sample_count, 176_000);
        assert!((info.duration_seconds - 11.0).abs() < 0.001);
        assert_eq!(info.format, "PCM WAV 16-bit");
    }

    #[test]
    fn writes_pcm16_wav_that_loads_back() {
        let audio = PcmAudio {
            path: PathBuf::from("memory.wav"),
            sample_rate: 16_000,
            channels: 1,
            samples: vec![-1.0, 0.0, 0.5],
        };
        let mut bytes = Vec::new();
        write_wav_pcm16_to(&mut bytes, &audio).unwrap();
        let loaded = load_wav_pcm_bytes("memory.wav", &bytes).unwrap();
        assert_eq!(loaded.sample_rate, 16_000);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.samples.len(), 3);
        assert!((loaded.samples[2] - 0.5).abs() < 0.0001);
    }
}
