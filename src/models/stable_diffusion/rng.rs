use super::{Result, SdTensor, StableDiffusionError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableDiffusionRng {
    state: u64,
}

impl StableDiffusionRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32_open01(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        ((mantissa as f32) + 0.5) / ((1u32 << 24) as f32)
    }

    pub fn next_standard_normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32_open01().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        (radius * theta.cos(), radius * theta.sin())
    }
}

pub fn deterministic_normal_tensor(shape: impl Into<Vec<usize>>, seed: u64) -> Result<SdTensor> {
    let shape = shape.into();
    let len = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| StableDiffusionError::InvalidInput("tensor shape overflow".to_string()))?;
    let mut rng = StableDiffusionRng::new(seed);
    let mut data = Vec::with_capacity(len);
    while data.len() < len {
        let (a, b) = rng.next_standard_normal_pair();
        data.push(a);
        if data.len() < len {
            data.push(b);
        }
    }
    SdTensor::new(shape, data)
}

pub fn deterministic_latents(
    seed: u64,
    batch: usize,
    channels: usize,
    height: u32,
    width: u32,
) -> Result<SdTensor> {
    deterministic_latents_with_scale(seed, batch, channels, height, width, 8)
}

pub fn deterministic_latents_with_scale(
    seed: u64,
    batch: usize,
    channels: usize,
    height: u32,
    width: u32,
    scale_factor: u32,
) -> Result<SdTensor> {
    if scale_factor == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "latent scale factor must be > 0".to_string(),
        ));
    }
    if height == 0 || width == 0 || height % scale_factor != 0 || width % scale_factor != 0 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "latent dimensions require positive image width and height multiples of {scale_factor}"
        )));
    }
    deterministic_normal_tensor(
        [
            batch,
            channels,
            (height / scale_factor) as usize,
            (width / scale_factor) as usize,
        ],
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_rng_replays_normal_tensor() {
        let first = deterministic_normal_tensor([2, 3], 42).unwrap();
        let second = deterministic_normal_tensor([2, 3], 42).unwrap();
        let different = deterministic_normal_tensor([2, 3], 43).unwrap();

        assert_eq!(first, second);
        assert_ne!(first, different);
        assert!(first.is_finite());
        let stats = first.stats().unwrap();
        assert!(stats.stddev > 0.0);
    }

    #[test]
    fn deterministic_latents_use_sd_spatial_scale() {
        let latents = deterministic_latents(7, 1, 4, 512, 512).unwrap();

        assert_eq!(latents.shape(), &[1, 4, 64, 64]);
        assert!(deterministic_latents(7, 1, 4, 510, 512).is_err());
    }

    #[test]
    fn deterministic_latents_support_custom_vae_scale() {
        let latents = deterministic_latents_with_scale(7, 1, 4, 64, 64, 2).unwrap();

        assert_eq!(latents.shape(), &[1, 4, 32, 32]);
    }
}
