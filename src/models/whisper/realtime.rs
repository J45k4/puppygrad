use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptionEvent {
    Partial {
        text: String,
    },
    Commit {
        text: String,
    },
    RawToken {
        phase: String,
        token_id: usize,
        token: String,
    },
    Silence,
    Warning {
        message: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptionJobPurpose {
    Partial,
    Final,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionJob {
    pub id: u64,
    pub stream_id: u64,
    pub window_start_seconds: f32,
    pub window_end_seconds: f32,
    pub samples: Vec<f32>,
    pub purpose: TranscriptionJobPurpose,
    pub max_new_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionDecodeResult {
    pub job_id: u64,
    pub text: String,
    pub generated_token_ids: Vec<usize>,
    pub no_speech_probability: f32,
    pub decode_millis: f32,
}

pub trait TranscriptionDecodeBackend {
    fn decode_one(&mut self, job: &TranscriptionJob) -> Result<TranscriptionDecodeResult, String>;

    fn decode_batch(
        &mut self,
        jobs: &[TranscriptionJob],
    ) -> Result<Vec<TranscriptionDecodeResult>, String> {
        jobs.iter().map(|job| self.decode_one(job)).collect()
    }
}

pub trait DecodeScheduler {
    fn decode(
        &mut self,
        jobs: Vec<TranscriptionJob>,
    ) -> Result<Vec<TranscriptionDecodeResult>, String>;
}

pub struct ImmediateDecodeScheduler<B> {
    backend: B,
}

impl<B> ImmediateDecodeScheduler<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B> DecodeScheduler for ImmediateDecodeScheduler<B>
where
    B: TranscriptionDecodeBackend,
{
    fn decode(
        &mut self,
        jobs: Vec<TranscriptionJob>,
    ) -> Result<Vec<TranscriptionDecodeResult>, String> {
        if jobs.len() == 1 {
            return self.backend.decode_one(&jobs[0]).map(|result| vec![result]);
        }
        self.backend.decode_batch(&jobs)
    }
}

#[derive(Clone, Debug)]
pub struct RollingAudioBuffer {
    samples: Vec<f32>,
    sample_rate: usize,
    max_samples: usize,
    start_sample_index: u64,
}

impl RollingAudioBuffer {
    pub fn new(sample_rate: usize, window_seconds: f32) -> Self {
        let max_samples = seconds_to_samples(sample_rate, window_seconds);
        Self {
            samples: Vec::new(),
            sample_rate,
            max_samples,
            start_sample_index: 0,
        }
    }

    pub fn append(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
        self.trim_to_window();
    }

    pub fn clear(&mut self) {
        self.start_sample_index += self.samples.len() as u64;
        self.samples.clear();
    }

    pub fn advance_samples(&mut self, sample_count: usize) {
        let drain = sample_count.min(self.samples.len());
        self.samples.drain(..drain);
        self.start_sample_index += drain as u64;
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn duration_seconds(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    pub fn window_start_seconds(&self) -> f32 {
        self.start_sample_index as f32 / self.sample_rate as f32
    }

    pub fn window_end_seconds(&self) -> f32 {
        self.window_start_seconds() + self.duration_seconds()
    }

    fn trim_to_window(&mut self) {
        if self.samples.len() <= self.max_samples {
            return;
        }
        let excess = self.samples.len() - self.max_samples;
        self.samples.drain(..excess);
        self.start_sample_index += excess as u64;
    }
}

#[derive(Clone, Debug)]
pub struct PartialCommitState {
    active_partial: Option<String>,
    stable_normalized: Option<String>,
    stable_count: usize,
    stable_decodes_to_commit: usize,
}

impl PartialCommitState {
    pub fn new(stable_decodes_to_commit: usize) -> Self {
        Self {
            active_partial: None,
            stable_normalized: None,
            stable_count: 0,
            stable_decodes_to_commit: stable_decodes_to_commit.max(1),
        }
    }

    pub fn observe_partial(&mut self, text: &str) -> PartialObservation {
        let normalized = normalize_transcript(text);
        if normalized.is_empty() {
            return PartialObservation::Empty;
        }
        let changed = self.active_partial.as_deref() != Some(text);
        self.active_partial = Some(text.to_string());
        if self.stable_normalized.as_deref() == Some(&normalized) {
            self.stable_count += 1;
        } else {
            self.stable_normalized = Some(normalized);
            self.stable_count = 1;
        }
        if self.stable_count >= self.stable_decodes_to_commit {
            PartialObservation::Commit(text.to_string())
        } else if changed {
            PartialObservation::Partial(text.to_string())
        } else {
            PartialObservation::Duplicate
        }
    }

    pub fn commit_active(&mut self) -> Option<String> {
        let text = self.active_partial.take()?;
        self.reset_stability();
        let normalized = normalize_transcript(&text);
        (!normalized.is_empty()).then_some(text)
    }

    pub fn clear(&mut self) {
        self.active_partial = None;
        self.reset_stability();
    }

    fn reset_stability(&mut self) {
        self.stable_normalized = None;
        self.stable_count = 0;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartialObservation {
    Empty,
    Duplicate,
    Partial(String),
    Commit(String),
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy = samples
        .iter()
        .map(|sample| {
            let value = *sample as f64;
            value * value
        })
        .sum::<f64>()
        / samples.len() as f64;
    energy.sqrt() as f32
}

pub fn is_silence(samples: &[f32], threshold: f32) -> bool {
    rms(samples) < threshold
}

pub fn normalize_transcript(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn seconds_to_samples(sample_rate: usize, seconds: f32) -> usize {
    ((sample_rate as f64 * seconds as f64).ceil() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_buffer_keeps_latest_window_and_advances() {
        let mut buffer = RollingAudioBuffer::new(10, 1.0);
        buffer.append(&[0.0; 6]);
        assert_eq!(buffer.samples().len(), 6);
        assert_eq!(buffer.window_start_seconds(), 0.0);

        buffer.append(&[1.0; 8]);
        assert_eq!(buffer.samples().len(), 10);
        assert_eq!(buffer.samples()[0], 0.0);
        assert_eq!(buffer.samples()[2], 1.0);
        assert!((buffer.window_start_seconds() - 0.4).abs() < 0.0001);

        buffer.advance_samples(3);
        assert_eq!(buffer.samples().len(), 7);
        assert!((buffer.window_start_seconds() - 0.7).abs() < 0.0001);

        buffer.clear();
        assert!(buffer.samples().is_empty());
        assert!((buffer.window_start_seconds() - 1.4).abs() < 0.0001);
    }

    #[test]
    fn detects_rms_silence() {
        assert!(is_silence(&[0.0, 0.001, -0.001], 0.01));
        assert!(!is_silence(&[0.2, -0.2], 0.01));
    }

    #[test]
    fn partial_state_suppresses_duplicates_and_commits_stable_text() {
        let mut state = PartialCommitState::new(3);
        assert_eq!(
            state.observe_partial(" Hello world "),
            PartialObservation::Partial(" Hello world ".to_string())
        );
        assert_eq!(
            state.observe_partial(" Hello world "),
            PartialObservation::Duplicate
        );
        assert_eq!(
            state.observe_partial("Hello  world"),
            PartialObservation::Commit("Hello  world".to_string())
        );
        assert_eq!(state.commit_active(), Some("Hello  world".to_string()));
        assert_eq!(state.observe_partial("   "), PartialObservation::Empty);
    }

    struct EchoBackend;

    impl TranscriptionDecodeBackend for EchoBackend {
        fn decode_one(
            &mut self,
            job: &TranscriptionJob,
        ) -> Result<TranscriptionDecodeResult, String> {
            Ok(TranscriptionDecodeResult {
                job_id: job.id,
                text: format!("{} samples", job.samples.len()),
                generated_token_ids: vec![job.max_new_tokens],
                no_speech_probability: 0.0,
                decode_millis: 1.0,
            })
        }
    }

    #[test]
    fn decode_backend_default_batch_routes_through_decode_one() {
        let mut backend = EchoBackend;
        let jobs = vec![
            TranscriptionJob {
                id: 1,
                stream_id: 7,
                window_start_seconds: 0.0,
                window_end_seconds: 1.0,
                samples: vec![0.0; 4],
                purpose: TranscriptionJobPurpose::Partial,
                max_new_tokens: 64,
            },
            TranscriptionJob {
                id: 2,
                stream_id: 7,
                window_start_seconds: 1.0,
                window_end_seconds: 2.0,
                samples: vec![0.0; 2],
                purpose: TranscriptionJobPurpose::Final,
                max_new_tokens: 32,
            },
        ];
        let results = backend.decode_batch(&jobs).unwrap();
        assert_eq!(results[0].job_id, 1);
        assert_eq!(results[0].text, "4 samples");
        assert_eq!(results[1].generated_token_ids, vec![32]);
    }

    #[test]
    fn immediate_scheduler_decodes_single_and_batch_jobs() {
        let mut scheduler = ImmediateDecodeScheduler::new(EchoBackend);
        let one = TranscriptionJob {
            id: 10,
            stream_id: 1,
            window_start_seconds: 0.0,
            window_end_seconds: 1.0,
            samples: vec![0.0; 3],
            purpose: TranscriptionJobPurpose::Partial,
            max_new_tokens: 8,
        };
        let two = TranscriptionJob {
            id: 11,
            max_new_tokens: 9,
            samples: vec![0.0; 5],
            ..one.clone()
        };

        let single = scheduler.decode(vec![one.clone()]).unwrap();
        assert_eq!(single[0].job_id, 10);

        let batch = scheduler.decode(vec![one, two]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[1].text, "5 samples");
    }
}
