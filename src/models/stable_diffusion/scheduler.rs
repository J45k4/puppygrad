use super::{DdimSchedulerConfig, Result, SdTensor, StableDiffusionError};

#[derive(Clone, Debug, PartialEq)]
pub struct DdimScheduler {
    pub config: DdimSchedulerConfig,
    pub betas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub timesteps: Vec<usize>,
}

impl DdimScheduler {
    pub fn new(config: DdimSchedulerConfig) -> Result<Self> {
        let betas = build_betas(&config)?;
        let mut product = 1.0f32;
        let alphas_cumprod = betas
            .iter()
            .map(|beta| {
                product *= 1.0 - beta;
                product
            })
            .collect::<Vec<_>>();
        Ok(Self {
            config,
            betas,
            alphas_cumprod,
            timesteps: Vec::new(),
        })
    }

    pub fn set_timesteps(&mut self, inference_steps: usize) -> Result<()> {
        if inference_steps == 0 {
            return Err(StableDiffusionError::InvalidInput(
                "inference step count must be > 0".to_string(),
            ));
        }
        if inference_steps > self.config.num_train_timesteps {
            return Err(StableDiffusionError::InvalidInput(format!(
                "inference step count {inference_steps} exceeds training timesteps {}",
                self.config.num_train_timesteps
            )));
        }
        let spacing = self.config.timestep_spacing.as_deref().unwrap_or("leading");
        let steps_offset = self.config.steps_offset.unwrap_or(0);
        self.timesteps = match spacing {
            "leading" => {
                let step_ratio = self.config.num_train_timesteps / inference_steps;
                (0..inference_steps)
                    .rev()
                    .map(|step| step * step_ratio + steps_offset)
                    .collect()
            }
            "trailing" => {
                let step_ratio = self.config.num_train_timesteps as f64 / inference_steps as f64;
                (0..inference_steps)
                    .map(|step| {
                        round_half_to_even(
                            self.config.num_train_timesteps as f64 - step as f64 * step_ratio,
                        ) as usize
                    })
                    .map(|step| step.saturating_sub(1))
                    .collect()
            }
            "linspace" => {
                if inference_steps == 1 {
                    vec![self.config.num_train_timesteps - 1]
                } else {
                    let max_timestep = (self.config.num_train_timesteps - 1) as f64;
                    (0..inference_steps)
                        .rev()
                        .map(|step| {
                            (step as f64 * max_timestep / (inference_steps - 1) as f64).round()
                                as usize
                        })
                        .collect()
                }
            }
            other => {
                return Err(StableDiffusionError::Unsupported(format!(
                    "scheduler timestep_spacing {other} is not supported"
                )))
            }
        };
        Ok(())
    }

    pub fn scale_model_input(&self, latents: &SdTensor, _timestep: usize) -> Result<SdTensor> {
        latents.scale(1.0)
    }

    pub fn step(
        &self,
        noise_pred: &SdTensor,
        timestep: usize,
        sample: &SdTensor,
    ) -> Result<SdTensor> {
        if noise_pred.shape() != sample.shape() {
            return Err(StableDiffusionError::InvalidInput(format!(
                "noise prediction shape {:?} does not match sample shape {:?}",
                noise_pred.shape(),
                sample.shape()
            )));
        }
        if timestep >= self.alphas_cumprod.len() {
            return Err(StableDiffusionError::InvalidInput(format!(
                "timestep {timestep} is out of range"
            )));
        }
        let step_ratio = if self.timesteps.len() > 1 {
            timestep.saturating_sub(next_lower_timestep(&self.timesteps, timestep))
        } else {
            self.config.num_train_timesteps
        };
        let prev_timestep = timestep.saturating_sub(step_ratio);
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if timestep == 0 {
            1.0
        } else {
            self.alphas_cumprod[prev_timestep]
        };
        let beta_prod_t = 1.0 - alpha_prod_t;

        let pred_original_sample = match self.config.prediction_type.as_str() {
            "epsilon" => sample
                .sub(&noise_pred.scale(beta_prod_t.sqrt())?)?
                .scale(1.0 / alpha_prod_t.sqrt())?,
            "v_prediction" => sample
                .scale(alpha_prod_t.sqrt())?
                .sub(&noise_pred.scale(beta_prod_t.sqrt())?)?,
            other => {
                return Err(StableDiffusionError::Unsupported(format!(
                    "scheduler prediction_type {other} is not implemented"
                )))
            }
        };
        let pred_original_sample = if self.config.clip_sample {
            pred_original_sample.clamp(-1.0, 1.0)?
        } else {
            pred_original_sample
        };
        let pred_sample_direction = noise_pred.scale((1.0 - alpha_prod_t_prev).sqrt())?;
        pred_original_sample
            .scale(alpha_prod_t_prev.sqrt())?
            .add(&pred_sample_direction)
    }
}

pub fn build_betas(config: &DdimSchedulerConfig) -> Result<Vec<f32>> {
    match config.beta_schedule.as_str() {
        "linear" => Ok(linspace(
            config.beta_start,
            config.beta_end,
            config.num_train_timesteps,
        )),
        "scaled_linear" => Ok(linspace(
            config.beta_start.sqrt(),
            config.beta_end.sqrt(),
            config.num_train_timesteps,
        )
        .into_iter()
        .map(|value| value * value)
        .collect()),
        "squaredcos_cap_v2" => Ok(betas_for_alpha_bar(config.num_train_timesteps, 0.999)),
        other => Err(StableDiffusionError::Unsupported(format!(
            "beta schedule {other} is not implemented"
        ))),
    }
}

fn betas_for_alpha_bar(num_train_timesteps: usize, max_beta: f32) -> Vec<f32> {
    fn alpha_bar(time_step: f32) -> f32 {
        let value = (time_step + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2;
        value.cos().powi(2)
    }

    (0..num_train_timesteps)
        .map(|index| {
            let t1 = index as f32 / num_train_timesteps as f32;
            let t2 = (index + 1) as f32 / num_train_timesteps as f32;
            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta)
        })
        .collect()
}

fn next_lower_timestep(timesteps: &[usize], timestep: usize) -> usize {
    timesteps
        .iter()
        .copied()
        .filter(|candidate| *candidate < timestep)
        .max()
        .unwrap_or(0)
}

fn round_half_to_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() < f64::EPSILON {
        if (floor as u64).is_multiple_of(2) {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start];
    }
    let denom = (steps - 1) as f32;
    (0..steps)
        .map(|index| start + (end - start) * index as f32 / denom)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DdimSchedulerConfig {
        DdimSchedulerConfig {
            class_name: "DDIMScheduler".to_string(),
            num_train_timesteps: 10,
            beta_start: 0.0001,
            beta_end: 0.02,
            beta_schedule: "linear".to_string(),
            clip_sample: false,
            prediction_type: "epsilon".to_string(),
            timestep_spacing: None,
            steps_offset: None,
        }
    }

    #[test]
    fn builds_descending_timesteps() {
        let mut scheduler = DdimScheduler::new(config()).unwrap();

        scheduler.set_timesteps(5).unwrap();

        assert_eq!(scheduler.timesteps, vec![8, 6, 4, 2, 0]);
    }

    #[test]
    fn supports_trailing_and_linspace_timesteps() {
        let mut trailing = config();
        trailing.timestep_spacing = Some("trailing".to_string());
        let mut scheduler = DdimScheduler::new(trailing).unwrap();
        scheduler.set_timesteps(4).unwrap();
        assert_eq!(scheduler.timesteps, vec![9, 7, 4, 1]);

        let mut linspace = config();
        linspace.timestep_spacing = Some("linspace".to_string());
        let mut scheduler = DdimScheduler::new(linspace).unwrap();
        scheduler.set_timesteps(4).unwrap();
        assert_eq!(scheduler.timesteps, vec![9, 6, 3, 0]);
    }

    #[test]
    fn builds_squared_cosine_betas() {
        let mut config = config();
        config.beta_schedule = "squaredcos_cap_v2".to_string();

        let betas = build_betas(&config).unwrap();

        assert_eq!(betas.len(), 10);
        assert!(betas.iter().all(|beta| beta.is_finite() && *beta > 0.0));
        assert!(betas.last().unwrap() <= &0.999);
    }

    #[test]
    fn one_step_preserves_shape_and_finiteness() {
        let mut scheduler = DdimScheduler::new(config()).unwrap();
        scheduler.set_timesteps(5).unwrap();
        let sample = SdTensor::new([1, 2], vec![0.5, -0.25]).unwrap();
        let noise = SdTensor::new([1, 2], vec![0.1, -0.2]).unwrap();

        let previous = scheduler
            .step(&noise, scheduler.timesteps[0], &sample)
            .unwrap();

        assert_eq!(previous.shape(), sample.shape());
        assert!(previous.is_finite());
    }
}
