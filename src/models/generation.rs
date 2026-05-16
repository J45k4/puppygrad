use clap::Args;
use std::error;
use std::fmt;
use std::time::Duration;

#[derive(Clone, Debug, Args, PartialEq)]
pub struct TextGenerationArgs {
    /// Max new tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub max_new_tokens: usize,

    /// Temperature (0 => greedy).
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Top-p nucleus sampling cutoff.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Top-k sampling cutoff.
    #[arg(long)]
    pub top_k: Option<usize>,

    /// RNG seed used when temperature is > 0.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// Repeat penalty (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    pub repeat_penalty: f32,

    /// How many recent tokens are considered for repeat penalty.
    #[arg(long, default_value_t = 128)]
    pub repeat_last_n: usize,
}

impl TextGenerationArgs {
    pub fn to_config(&self) -> TextGenerationConfig {
        TextGenerationConfig {
            max_new_tokens: self.max_new_tokens,
            eos_token_id: None,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextGenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_id: Option<usize>,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl TextGenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            eos_token_id: None,
            temperature: 0.0,
            top_p: None,
            top_k: None,
            seed: 299_792_458,
            repeat_penalty: 1.0,
            repeat_last_n: 128,
        }
    }

    pub fn with_eos_token_id(mut self, eos_token_id: Option<usize>) -> Self {
        self.eos_token_id = eos_token_id;
        self
    }

    pub fn validate(&self) -> Result<(), GenerationConfigError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GenerationConfigError(
                "temperature must be finite and >= 0".to_string(),
            ));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(GenerationConfigError(
                    "top_p must be finite and in (0, 1]".to_string(),
                ));
            }
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(GenerationConfigError("top_k must be > 0".to_string()));
            }
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(GenerationConfigError(
                "repeat_penalty must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationConfigError(String);

impl fmt::Display for GenerationConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl error::Error for GenerationConfigError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokenize_time: Duration,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub total_generation_time: Duration,
    pub first_token_time: Option<Duration>,
}

impl GenerationStats {
    pub fn with_profile<P>(self, profile: P) -> ProfiledGenerationStats<P> {
        ProfiledGenerationStats {
            common: self,
            profile,
        }
    }

    pub fn total_model_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_time)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.decode_time)
    }

    pub fn total_tokens_per_second(&self) -> f64 {
        rate(self.total_model_tokens(), self.total_generation_time)
    }

    pub fn average_decode_token_time(&self) -> Option<Duration> {
        if self.generated_tokens == 0 {
            return None;
        }
        Some(Duration::from_secs_f64(
            self.decode_time.as_secs_f64() / self.generated_tokens as f64,
        ))
    }
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }
    tokens as f64 / seconds
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfiledGenerationStats<P> {
    pub common: GenerationStats,
    pub profile: P,
}

impl<P> ProfiledGenerationStats<P> {
    pub fn new(common: GenerationStats, profile: P) -> Self {
        Self { common, profile }
    }

    pub fn split(self) -> (GenerationStats, P) {
        (self.common, self.profile)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplingError(String);

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl error::Error for SamplingError {}

#[derive(Clone, Debug)]
pub struct LogitsSampler {
    rng: SmallRng,
}

impl LogitsSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::new(seed),
        }
    }

    pub fn select_next_token(
        &mut self,
        logits: &[f32],
        history: &[usize],
        config: &TextGenerationConfig,
    ) -> Result<usize, SamplingError> {
        let mut adjusted = logits.to_vec();
        apply_repeat_penalty(
            &mut adjusted,
            history,
            config.repeat_penalty,
            config.repeat_last_n,
        );

        if config.temperature <= 0.0 {
            return Ok(argmax_logits(&adjusted));
        }

        sample_logits(&adjusted, config, &mut self.rng)
    }
}

#[derive(Clone, Debug)]
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

pub fn argmax_logits(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}

fn apply_repeat_penalty(logits: &mut [f32], history: &[usize], penalty: f32, last_n: usize) {
    if penalty == 1.0 || last_n == 0 {
        return;
    }

    let mut seen = Vec::new();
    for token_id in history.iter().rev().take(last_n).copied() {
        if token_id >= logits.len() || seen.contains(&token_id) {
            continue;
        }
        seen.push(token_id);
        if logits[token_id] < 0.0 {
            logits[token_id] *= penalty;
        } else {
            logits[token_id] /= penalty;
        }
    }
}

fn sample_logits(
    logits: &[f32],
    config: &TextGenerationConfig,
    rng: &mut SmallRng,
) -> Result<usize, SamplingError> {
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .collect();
    if candidates.is_empty() {
        return Err(SamplingError(
            "cannot sample from empty or non-finite logits".to_string(),
        ));
    }

    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if let Some(top_k) = config.top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }

    let max_logit = candidates[0].1 as f64 / config.temperature as f64;
    let mut weighted: Vec<(usize, f64)> = candidates
        .into_iter()
        .map(|(token_id, logit)| {
            let scaled = logit as f64 / config.temperature as f64;
            (token_id, (scaled - max_logit).exp())
        })
        .collect();

    if let Some(top_p) = config.top_p {
        let full_total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
        if full_total > 0.0 {
            let mut cumulative = 0.0;
            let mut keep_len = 0;
            for (_, weight) in &weighted {
                cumulative += *weight;
                keep_len += 1;
                if cumulative / full_total >= top_p as f64 {
                    break;
                }
            }
            weighted.truncate(keep_len.max(1));
        }
    }

    let total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
    if total <= 0.0 || !total.is_finite() {
        return Ok(weighted[0].0);
    }

    let fallback = weighted.last().map(|(token_id, _)| *token_id).unwrap();
    let mut target = rng.next_unit_f64() * total;
    for (token_id, weight) in weighted {
        target -= weight;
        if target <= 0.0 {
            return Ok(token_id);
        }
    }

    Ok(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_penalty_can_change_greedy_selection() -> Result<(), SamplingError> {
        let config = TextGenerationConfig {
            repeat_penalty: 2.0,
            ..TextGenerationConfig::new(1)
        };
        let mut sampler = LogitsSampler::new(config.seed);

        let token = sampler.select_next_token(&[1.0, 3.0, 2.0], &[1], &config)?;

        assert_eq!(token, 2);
        Ok(())
    }

    #[test]
    fn top_k_one_samples_best_candidate() -> Result<(), SamplingError> {
        let config = TextGenerationConfig {
            temperature: 1.0,
            top_k: Some(1),
            ..TextGenerationConfig::new(1)
        };
        let mut sampler = LogitsSampler::new(config.seed);

        let token = sampler.select_next_token(&[0.0, 10.0, 9.0], &[], &config)?;

        assert_eq!(token, 1);
        Ok(())
    }

    #[test]
    fn validates_sampling_config() {
        let mut config = TextGenerationConfig::new(1);
        config.top_p = Some(2.0);

        assert!(config.validate().is_err());
    }
}
