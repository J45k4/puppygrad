use super::{
    classifier_free_guidance, deterministic_latents_with_scale, diffusers_decoded_to_rgb,
    load_clip_text_config, load_clip_text_weights, load_model_index, load_safetensors_manifest,
    load_scheduler_config, load_unet_2d_condition_model_weights, load_unet_config, load_vae_config,
    load_vae_decoder_model_weights, save_rgb_tensor_image, unet_forward, vae_decode_latents,
    AutoencoderKlConfig, ClipTextConfig, ClipTextEncoder, DdimScheduler, Result,
    StableDiffusionAssetPaths, StableDiffusionError, StableDiffusionGenerationMetadata,
    StableDiffusionRuntimeOptions, StableDiffusionScheduler, StableDiffusionTensorManifestRow,
    StableDiffusionTokenizer, Unet2DConditionConfig, Unet2DConditionModelWeights,
    VaeDecoderModelWeights,
};

#[derive(Clone)]
pub struct StableDiffusionPipeline {
    pub paths: StableDiffusionAssetPaths,
    pub tokenizer: StableDiffusionTokenizer,
    pub clip_config: ClipTextConfig,
    pub clip_encoder: ClipTextEncoder,
    pub unet_config: Unet2DConditionConfig,
    pub unet_weights: Unet2DConditionModelWeights,
    pub vae_config: AutoencoderKlConfig,
    pub vae_weights: VaeDecoderModelWeights,
    pub scheduler: DdimScheduler,
    pub unet_manifest: Vec<StableDiffusionTensorManifestRow>,
    pub vae_manifest: Vec<StableDiffusionTensorManifestRow>,
}

impl StableDiffusionPipeline {
    pub fn from_assets(paths: StableDiffusionAssetPaths) -> Result<Self> {
        let _model_index = load_model_index(&paths.model_index)?;
        let clip_config = load_clip_text_config(&paths.text_encoder_config)?;
        let clip_weights = load_clip_text_weights(&paths.text_encoder_safetensors, &clip_config)?;
        let unet_config = load_unet_config(&paths.unet_config)?;
        let unet_weights =
            load_unet_2d_condition_model_weights(&paths.unet_safetensors, &unet_config)?;
        let vae_config = load_vae_config(&paths.vae_config)?;
        let vae_weights = load_vae_decoder_model_weights(&paths.vae_safetensors, &vae_config)?;
        let unet_manifest = load_safetensors_manifest(&paths.unet_safetensors)?;
        let vae_manifest = load_safetensors_manifest(&paths.vae_safetensors)?;
        let scheduler = DdimScheduler::new(load_scheduler_config(&paths.scheduler_config)?)?;
        let tokenizer = StableDiffusionTokenizer::from_diffusers_files(
            &paths.tokenizer_json,
            &paths.tokenizer_vocab,
            &paths.tokenizer_merges,
        )?;
        let clip_encoder = ClipTextEncoder::new(clip_config.clone(), clip_weights)?;
        Ok(Self {
            paths,
            tokenizer,
            clip_config,
            clip_encoder,
            unet_config,
            unet_weights,
            vae_config,
            vae_weights,
            scheduler,
            unet_manifest,
            vae_manifest,
        })
    }

    pub fn generate(
        &self,
        options: &StableDiffusionRuntimeOptions,
    ) -> Result<StableDiffusionGenerationMetadata> {
        if options.scheduler != StableDiffusionScheduler::Ddim {
            return Err(StableDiffusionError::Unsupported(format!(
                "native Rust backend currently supports only DDIM scheduler, got {}",
                options.scheduler.label()
            )));
        }

        let conditioning = self
            .tokenizer
            .encode_conditioning(&options.prompt, &options.negative_prompt)?;
        let prompt_embeddings = self
            .clip_encoder
            .encode_token_ids(&conditioning.prompt.token_ids)?;
        let negative_embeddings = self
            .clip_encoder
            .encode_token_ids(&conditioning.negative_prompt.token_ids)?;
        let vae_scale_factor = 1u32
            .checked_shl(self.vae_config.block_out_channels.len().saturating_sub(1) as u32)
            .ok_or_else(|| {
                StableDiffusionError::Config("VAE scale factor overflowed u32".to_string())
            })?;
        let mut latents = deterministic_latents_with_scale(
            options.seed,
            1,
            self.vae_config.latent_channels,
            options.height,
            options.width,
            vae_scale_factor,
        )?;
        let mut scheduler = self.scheduler.clone();
        scheduler.set_timesteps(options.steps)?;

        let started = std::time::Instant::now();
        eprintln!(
            "stable-diffusion progress: encoded prompts; denoising {}/{} steps",
            0,
            scheduler.timesteps.len()
        );
        for (step_index, timestep) in scheduler.timesteps.clone().into_iter().enumerate() {
            let latent_model_input = scheduler.scale_model_input(&latents, timestep)?;
            let noise_uncond = unet_forward(
                &latent_model_input,
                timestep,
                &negative_embeddings,
                &self.unet_weights,
            )?;
            let noise_cond = unet_forward(
                &latent_model_input,
                timestep,
                &prompt_embeddings,
                &self.unet_weights,
            )?;
            let noise =
                classifier_free_guidance(&noise_uncond, &noise_cond, options.guidance_scale)?;
            latents = scheduler.step(&noise, timestep, &latents)?;
            if options.stats {
                let stats = latents.stats()?;
                eprintln!(
                    "stable-diffusion stats: step={}/{} timestep={} latent_min={:.6} latent_max={:.6} latent_mean={:.6} latent_std={:.6} latent_rms={:.6} elapsed={:.3}s",
                    step_index + 1,
                    scheduler.timesteps.len(),
                    timestep,
                    stats.min,
                    stats.max,
                    stats.mean,
                    stats.stddev,
                    stats.rms,
                    started.elapsed().as_secs_f64()
                );
            } else {
                eprintln!(
                    "stable-diffusion progress: step={}/{} timestep={} elapsed={:.3}s",
                    step_index + 1,
                    scheduler.timesteps.len(),
                    timestep,
                    started.elapsed().as_secs_f64()
                );
            }
        }

        eprintln!("stable-diffusion progress: decoding latents with VAE");
        let decoded = vae_decode_latents(&latents, &self.vae_config, &self.vae_weights)?;
        let rgb = diffusers_decoded_to_rgb(&decoded)?;
        if options.stats {
            let decoded_stats = decoded.stats()?;
            let rgb_stats = rgb.stats()?;
            eprintln!(
                "stable-diffusion stats: decoded_shape={:?} decoded_min={:.6} decoded_max={:.6} decoded_mean={:.6} decoded_std={:.6} rgb_min={:.6} rgb_max={:.6} rgb_mean={:.6} rgb_std={:.6}",
                decoded.shape(),
                decoded_stats.min,
                decoded_stats.max,
                decoded_stats.mean,
                decoded_stats.stddev,
                rgb_stats.min,
                rgb_stats.max,
                rgb_stats.mean,
                rgb_stats.stddev
            );
        }
        eprintln!(
            "stable-diffusion progress: saving {} image to {}",
            options.output_format.label(),
            options.out.display()
        );
        save_rgb_tensor_image(
            &rgb,
            &options.out,
            options.width,
            options.height,
            options.output_format,
        )?;

        Ok(StableDiffusionGenerationMetadata {
            backend: options.backend,
            model_source: self.paths.model_dir.display().to_string(),
            width: options.width,
            height: options.height,
            seed: options.seed,
            steps: options.steps,
            scheduler: options.scheduler,
            guidance_scale: options.guidance_scale,
            output_path: options.out.clone(),
            elapsed: Some(started.elapsed()),
        })
    }
}
