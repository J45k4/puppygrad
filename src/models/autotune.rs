#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoTuneOptions {
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub max_trials: Option<usize>,
}

impl Default for AutoTuneOptions {
    fn default() -> Self {
        Self {
            warmup_runs: 1,
            measured_runs: 3,
            max_trials: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoTuneTrial<C, M> {
    pub config: C,
    pub measurement: M,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoTuneResult<C, M> {
    pub best_config: C,
    pub best_score: f64,
    pub trials: Vec<AutoTuneTrial<C, M>>,
}

pub trait AutoTuneTarget {
    type Config: Clone;
    type Measurement;
    type Error;

    fn candidate_configs(&self) -> Vec<Self::Config>;

    fn evaluate_config(
        &mut self,
        config: &Self::Config,
        options: &AutoTuneOptions,
    ) -> Result<Self::Measurement, Self::Error>;

    fn score(&self, measurement: &Self::Measurement) -> f64;
}

pub fn autotune<T>(
    target: &mut T,
    options: &AutoTuneOptions,
) -> Result<AutoTuneResult<T::Config, T::Measurement>, T::Error>
where
    T: AutoTuneTarget,
{
    let mut candidates = target.candidate_configs();
    if let Some(max_trials) = options.max_trials {
        candidates.truncate(max_trials);
    }

    let mut trials = Vec::with_capacity(candidates.len());
    let mut best_index = None;
    let mut best_score = f64::NEG_INFINITY;

    for config in candidates {
        let measurement = target.evaluate_config(&config, options)?;
        let score = target.score(&measurement);
        if score > best_score {
            best_score = score;
            best_index = Some(trials.len());
        }
        trials.push(AutoTuneTrial {
            config,
            measurement,
            score,
        });
    }

    let best_index = best_index.expect("autotune requires at least one candidate");
    let best_config = trials[best_index].config.clone();
    Ok(AutoTuneResult {
        best_config,
        best_score,
        trials,
    })
}
