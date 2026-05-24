use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{Result, StableDiffusionError};

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
use super::{
    classifier_free_guidance, diffusers_decoded_to_rgb, load_clip_text_config,
    load_clip_text_weights, load_scheduler_config, load_unet_2d_condition_model_weights,
    load_unet_config, load_vae_config, load_vae_decoder_model_weights, unet_forward,
    vae_decode_latents, ClipTextEncoder, DdimScheduler, SdTensor, StableDiffusionAssetPaths,
    StableDiffusionTokenizer,
};

#[cfg(test)]
const FIXTURE_DIR: &str = "tests/data/stable_diffusion";
#[cfg(test)]
const PARITY_ABS_TOLERANCE: f32 = 5.0e-2;
#[cfg(test)]
const PARITY_REL_TOLERANCE: f32 = 5.0e-2;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionReferenceFixture {
    pub metadata: StableDiffusionReferenceMetadata,
    pub tokenizer: StableDiffusionTokenizerReference,
    #[serde(default)]
    pub scheduler: Option<StableDiffusionSchedulerReference>,
    #[serde(default)]
    pub clip: Option<StableDiffusionClipReference>,
    #[serde(default)]
    pub latents: Option<StableDiffusionLatentsReference>,
    #[serde(default)]
    pub unet: Option<StableDiffusionUnetReference>,
    #[serde(default)]
    pub vae: Option<StableDiffusionVaeReference>,
    #[serde(default)]
    pub image: Option<StableDiffusionImageReference>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionReferenceMetadata {
    pub model_source: String,
    #[serde(default)]
    pub revision: Option<String>,
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub scheduler: String,
    pub steps: usize,
    pub guidance_scale: f32,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub diffusers_version: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
    #[serde(default)]
    pub torch_version: Option<String>,
    #[serde(default)]
    pub python_version: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct StableDiffusionTokenizerReference {
    pub conditional_token_ids: Vec<u32>,
    pub unconditional_token_ids: Vec<u32>,
    #[serde(default)]
    pub conditional_attention_mask: Vec<u32>,
    #[serde(default)]
    pub unconditional_attention_mask: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionSchedulerReference {
    pub timesteps: Vec<i64>,
    #[serde(default)]
    pub init_noise_sigma: Option<f32>,
    #[serde(default)]
    pub alphas_cumprod_head: Vec<f32>,
    #[serde(default)]
    pub alphas_cumprod_tail: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionClipReference {
    pub conditional_embedding: StableDiffusionTensorReference,
    pub unconditional_embedding: StableDiffusionTensorReference,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionLatentsReference {
    pub initial: StableDiffusionTensorReference,
    #[serde(default)]
    pub first_step: Option<StableDiffusionTensorReference>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionUnetReference {
    pub first_timestep: usize,
    pub latent_model_input: StableDiffusionTensorReference,
    pub unconditional_noise: StableDiffusionTensorReference,
    pub conditional_noise: StableDiffusionTensorReference,
    pub guided_noise: StableDiffusionTensorReference,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionVaeReference {
    pub scaling_factor: f32,
    pub decoded: StableDiffusionTensorReference,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionImageReference {
    pub rgb: StableDiffusionTensorReference,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionTensorReference {
    pub shape: Vec<usize>,
    pub slice: Vec<f32>,
    #[serde(default)]
    pub values: Vec<f32>,
    #[serde(default)]
    pub stats: Option<StableDiffusionTensorStatsReference>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct StableDiffusionTensorStatsReference {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    #[serde(rename = "std")]
    pub stddev: f32,
    pub rms: f32,
}

pub fn load_stable_diffusion_reference_fixture(
    path: impl AsRef<Path>,
) -> Result<StableDiffusionReferenceFixture> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|err| {
        StableDiffusionError::Asset(format!(
            "failed to read Stable Diffusion fixture {}: {err}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        StableDiffusionError::Asset(format!(
            "failed to parse Stable Diffusion fixture {}: {err}",
            path.display()
        ))
    })
}

#[cfg(test)]
fn fixture_paths() -> Vec<PathBuf> {
    let mut paths = std::env::var_os("PUPPYGRAD_SD_PARITY_FIXTURE")
        .or_else(|| std::env::var_os("SD_PARITY_FIXTURE"))
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();

    let dir = Path::new(FIXTURE_DIR);
    if let Ok(entries) = fs::read_dir(dir) {
        paths.extend(
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.extension()
                        .is_some_and(|extension| extension == "json")
                }),
        );
    }
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(test)]
fn all_fixtures() -> Result<Vec<StableDiffusionReferenceFixture>> {
    fixture_paths()
        .iter()
        .map(load_stable_diffusion_reference_fixture)
        .collect()
}

#[cfg(test)]
fn fixture_model_paths(
    fixture: &StableDiffusionReferenceFixture,
) -> Option<StableDiffusionAssetPaths> {
    std::env::var_os("PUPPYGRAD_SD_PARITY_MODEL_DIR")
        .or_else(|| std::env::var_os("SD_PARITY_MODEL_DIR"))
        .map(PathBuf::from)
        .or_else(|| {
            let path = PathBuf::from(&fixture.metadata.model_source);
            path.is_dir().then_some(path)
        })
        .map(StableDiffusionAssetPaths::new)
}

#[cfg(test)]
fn assert_close(actual: f32, expected: f32, label: &str) {
    let diff = (actual - expected).abs();
    let allowed = PARITY_ABS_TOLERANCE.max(expected.abs() * PARITY_REL_TOLERANCE);
    assert!(
        diff <= allowed,
        "{label}: actual {actual} expected {expected} diff {diff} allowed {allowed}"
    );
}

#[cfg(test)]
fn assert_tensor_matches_reference(
    actual: &SdTensor,
    expected: &StableDiffusionTensorReference,
    label: &str,
) -> Result<()> {
    assert_eq!(actual.shape(), expected.shape.as_slice(), "{label} shape");
    assert!(
        expected.slice.len() <= actual.len(),
        "{label} reference slice length {} exceeds actual tensor length {}",
        expected.slice.len(),
        actual.len()
    );
    let print_stats = std::env::var_os("PUPPYGRAD_SD_PARITY_PRINT_STATS").is_some();
    if print_stats {
        if let Some(expected_stats) = expected.stats {
            let actual_stats = actual.stats()?;
            eprintln!(
                "{label} stats: actual min={:.6} max={:.6} mean={:.6} std={:.6} rms={:.6}; expected min={:.6} max={:.6} mean={:.6} std={:.6} rms={:.6}",
                actual_stats.min,
                actual_stats.max,
                actual_stats.mean,
                actual_stats.stddev,
                actual_stats.rms,
                expected_stats.min,
                expected_stats.max,
                expected_stats.mean,
                expected_stats.stddev,
                expected_stats.rms,
            );
        }
    }
    for (index, expected_value) in expected.slice.iter().copied().enumerate() {
        assert_close(
            actual.data()[index],
            expected_value,
            &format!("{label} slice[{index}]"),
        );
    }
    if let Some(expected_stats) = expected.stats {
        let actual_stats = actual.stats()?;
        assert_close(
            actual_stats.min,
            expected_stats.min,
            &format!("{label} min"),
        );
        assert_close(
            actual_stats.max,
            expected_stats.max,
            &format!("{label} max"),
        );
        assert_close(
            actual_stats.mean,
            expected_stats.mean,
            &format!("{label} mean"),
        );
        assert_close(
            actual_stats.stddev,
            expected_stats.stddev,
            &format!("{label} stddev"),
        );
        assert_close(
            actual_stats.rms,
            expected_stats.rms,
            &format!("{label} rms"),
        );
    }
    Ok(())
}

#[cfg(test)]
fn tensor_from_reference(reference: &StableDiffusionTensorReference) -> Option<Result<SdTensor>> {
    (!reference.values.is_empty())
        .then(|| SdTensor::new(reference.shape.clone(), reference.values.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_compact_reference_fixture() -> Result<()> {
        let fixture: StableDiffusionReferenceFixture = serde_json::from_str(
            r#"{
                "metadata": {
                    "model_source": "hf-internal-testing/tiny-stable-diffusion-pipe",
                    "revision": "main",
                    "prompt": "hello",
                    "negative_prompt": "",
                    "seed": 42,
                    "scheduler": "ddim",
                    "steps": 2,
                    "guidance_scale": 7.5,
                    "width": 64,
                    "height": 64,
                    "diffusers_version": "0.0.0",
                    "transformers_version": "0.0.0",
                    "torch_version": "0.0.0",
                    "python_version": "3.x"
                },
                "tokenizer": {
                    "conditional_token_ids": [1, 2, 3],
                    "unconditional_token_ids": [1, 3],
                    "conditional_attention_mask": [1, 1, 1],
                    "unconditional_attention_mask": [1, 1]
                },
                "scheduler": {
                    "timesteps": [999, 499],
                    "init_noise_sigma": 1.0,
                    "alphas_cumprod_head": [0.99],
                    "alphas_cumprod_tail": [0.01]
                }
            }"#,
        )
        .map_err(|err| StableDiffusionError::Asset(err.to_string()))?;

        assert_eq!(fixture.metadata.prompt, "hello");
        assert_eq!(fixture.tokenizer.conditional_token_ids, vec![1, 2, 3]);
        assert_eq!(
            fixture.scheduler.as_ref().unwrap().timesteps,
            vec![999, 499]
        );
        Ok(())
    }

    #[test]
    fn tokenizer_reference_fixtures_match_when_model_assets_available() -> Result<()> {
        for fixture in all_fixtures()? {
            let Some(paths) = fixture_model_paths(&fixture) else {
                continue;
            };
            if !paths.tokenizer_vocab.exists() || !paths.tokenizer_merges.exists() {
                continue;
            }
            let tokenizer = StableDiffusionTokenizer::from_diffusers_files(
                &paths.tokenizer_json,
                &paths.tokenizer_vocab,
                &paths.tokenizer_merges,
            )?;
            let conditioning = tokenizer
                .encode_conditioning(&fixture.metadata.prompt, &fixture.metadata.negative_prompt)?;

            assert_eq!(
                conditioning.prompt.token_ids,
                fixture.tokenizer.conditional_token_ids
            );
            assert_eq!(
                conditioning.negative_prompt.token_ids,
                fixture.tokenizer.unconditional_token_ids
            );
            if !fixture.tokenizer.conditional_attention_mask.is_empty() {
                assert_eq!(
                    conditioning.prompt.attention_mask,
                    fixture.tokenizer.conditional_attention_mask
                );
            }
            if !fixture.tokenizer.unconditional_attention_mask.is_empty() {
                assert_eq!(
                    conditioning.negative_prompt.attention_mask,
                    fixture.tokenizer.unconditional_attention_mask
                );
            }
        }
        Ok(())
    }

    #[test]
    fn clip_reference_fixtures_match_when_model_assets_available() -> Result<()> {
        for fixture in all_fixtures()? {
            let Some(reference) = &fixture.clip else {
                continue;
            };
            let Some(paths) = fixture_model_paths(&fixture) else {
                continue;
            };
            if !paths.text_encoder_config.exists()
                || !paths.text_encoder_safetensors.exists()
                || !paths.tokenizer_vocab.exists()
                || !paths.tokenizer_merges.exists()
            {
                continue;
            }
            let tokenizer = StableDiffusionTokenizer::from_diffusers_files(
                &paths.tokenizer_json,
                &paths.tokenizer_vocab,
                &paths.tokenizer_merges,
            )?;
            let conditioning = tokenizer
                .encode_conditioning(&fixture.metadata.prompt, &fixture.metadata.negative_prompt)?;
            let config = load_clip_text_config(&paths.text_encoder_config)?;
            let weights = load_clip_text_weights(&paths.text_encoder_safetensors, &config)?;
            let encoder = ClipTextEncoder::new(config, weights)?;
            let conditional = encoder.encode_token_ids(&conditioning.prompt.token_ids)?;
            let unconditional =
                encoder.encode_token_ids(&conditioning.negative_prompt.token_ids)?;

            assert_tensor_matches_reference(
                &conditional,
                &reference.conditional_embedding,
                "conditional CLIP embedding",
            )?;
            assert_tensor_matches_reference(
                &unconditional,
                &reference.unconditional_embedding,
                "unconditional CLIP embedding",
            )?;
        }
        Ok(())
    }

    #[test]
    fn scheduler_reference_fixtures_match_when_model_assets_available() -> Result<()> {
        for fixture in all_fixtures()? {
            if fixture.metadata.scheduler != "ddim" {
                continue;
            }
            let Some(reference) = &fixture.scheduler else {
                continue;
            };
            let Some(paths) = fixture_model_paths(&fixture) else {
                continue;
            };
            if !paths.scheduler_config.exists() {
                continue;
            }
            let mut scheduler =
                DdimScheduler::new(load_scheduler_config(&paths.scheduler_config)?)?;
            scheduler.set_timesteps(fixture.metadata.steps)?;

            let timesteps = scheduler
                .timesteps
                .iter()
                .map(|value| *value as i64)
                .collect::<Vec<_>>();
            assert_eq!(timesteps, reference.timesteps);
            for (index, expected) in reference.alphas_cumprod_head.iter().copied().enumerate() {
                assert_close(
                    scheduler.alphas_cumprod[index],
                    expected,
                    &format!("scheduler alpha head {index}"),
                );
            }
            let tail_start = scheduler
                .alphas_cumprod
                .len()
                .saturating_sub(reference.alphas_cumprod_tail.len());
            for (index, expected) in reference.alphas_cumprod_tail.iter().copied().enumerate() {
                assert_close(
                    scheduler.alphas_cumprod[tail_start + index],
                    expected,
                    &format!("scheduler alpha tail {index}"),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn unet_and_vae_reference_fixtures_match_when_model_assets_available() -> Result<()> {
        for fixture in all_fixtures()? {
            if fixture.metadata.scheduler != "ddim" {
                continue;
            }
            let (Some(latents), Some(unet_reference)) = (&fixture.latents, &fixture.unet) else {
                continue;
            };
            let Some(initial_latents) = tensor_from_reference(&latents.initial).transpose()? else {
                continue;
            };
            let Some(paths) = fixture_model_paths(&fixture) else {
                continue;
            };
            if !paths.unet_config.exists()
                || !paths.unet_safetensors.exists()
                || !paths.vae_config.exists()
                || !paths.vae_safetensors.exists()
                || !paths.scheduler_config.exists()
                || !paths.text_encoder_config.exists()
                || !paths.text_encoder_safetensors.exists()
                || !paths.tokenizer_vocab.exists()
                || !paths.tokenizer_merges.exists()
            {
                continue;
            }

            let tokenizer = StableDiffusionTokenizer::from_diffusers_files(
                &paths.tokenizer_json,
                &paths.tokenizer_vocab,
                &paths.tokenizer_merges,
            )?;
            let conditioning = tokenizer
                .encode_conditioning(&fixture.metadata.prompt, &fixture.metadata.negative_prompt)?;
            let clip_config = load_clip_text_config(&paths.text_encoder_config)?;
            let clip_weights =
                load_clip_text_weights(&paths.text_encoder_safetensors, &clip_config)?;
            let clip_encoder = ClipTextEncoder::new(clip_config, clip_weights)?;
            let prompt_embeddings =
                clip_encoder.encode_token_ids(&conditioning.prompt.token_ids)?;
            let negative_embeddings =
                clip_encoder.encode_token_ids(&conditioning.negative_prompt.token_ids)?;

            let mut scheduler =
                DdimScheduler::new(load_scheduler_config(&paths.scheduler_config)?)?;
            scheduler.set_timesteps(fixture.metadata.steps)?;
            let timestep = scheduler.timesteps[0];
            assert_eq!(timestep, unet_reference.first_timestep);
            let latent_model_input = scheduler.scale_model_input(&initial_latents, timestep)?;
            assert_tensor_matches_reference(
                &latent_model_input,
                &unet_reference.latent_model_input,
                "UNet latent input",
            )?;

            let unet_config = load_unet_config(&paths.unet_config)?;
            let unet_weights =
                load_unet_2d_condition_model_weights(&paths.unet_safetensors, &unet_config)?;
            let noise_uncond = unet_forward(
                &latent_model_input,
                timestep,
                &negative_embeddings,
                &unet_weights,
            )?;
            let noise_cond = unet_forward(
                &latent_model_input,
                timestep,
                &prompt_embeddings,
                &unet_weights,
            )?;
            let guided = classifier_free_guidance(
                &noise_uncond,
                &noise_cond,
                fixture.metadata.guidance_scale,
            )?;

            assert_tensor_matches_reference(
                &noise_uncond,
                &unet_reference.unconditional_noise,
                "UNet unconditional noise",
            )?;
            assert_tensor_matches_reference(
                &noise_cond,
                &unet_reference.conditional_noise,
                "UNet conditional noise",
            )?;
            assert_tensor_matches_reference(&guided, &unet_reference.guided_noise, "guided noise")?;

            let first_step_latents = scheduler.step(&guided, timestep, &initial_latents)?;
            if let Some(reference) = &latents.first_step {
                assert_tensor_matches_reference(
                    &first_step_latents,
                    reference,
                    "first step latents",
                )?;
            }
            if let Some(vae_reference) = &fixture.vae {
                let vae_config = load_vae_config(&paths.vae_config)?;
                let vae_weights =
                    load_vae_decoder_model_weights(&paths.vae_safetensors, &vae_config)?;
                let decoded = vae_decode_latents(&first_step_latents, &vae_config, &vae_weights)?;
                assert_tensor_matches_reference(&decoded, &vae_reference.decoded, "VAE decoded")?;
                if let Some(image_reference) = &fixture.image {
                    let rgb = diffusers_decoded_to_rgb(&decoded)?;
                    assert_tensor_matches_reference(
                        &rgb,
                        &image_reference.rgb,
                        "RGB image tensor",
                    )?;
                }
            }
        }
        Ok(())
    }
}
