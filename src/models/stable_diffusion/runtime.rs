use std::path::PathBuf;
use std::time::Duration;

use super::{
    generate_stable_diffusion_python, prepare_stable_diffusion_assets,
    validate_stable_diffusion_native_assets, Result, StableDiffusionAssetPaths,
    StableDiffusionError, StableDiffusionPipeline, STABLE_DIFFUSION_V1_5_MODEL_ID,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StableDiffusionBackend {
    PythonDiffusers,
    Rust,
}

impl StableDiffusionBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::PythonDiffusers => "python-diffusers",
            Self::Rust => "rust",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StableDiffusionScheduler {
    Ddim,
    Euler,
    Ddpm,
}

impl StableDiffusionScheduler {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ddim => "ddim",
            Self::Euler => "euler",
            Self::Ddpm => "ddpm",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StableDiffusionOutputFormat {
    Png,
    Jpeg,
}

impl StableDiffusionOutputFormat {
    pub fn infer(path: &std::path::Path) -> Result<Self> {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("png") => Ok(Self::Png),
            Some("jpg" | "jpeg") => Ok(Self::Jpeg),
            Some(extension) => Err(StableDiffusionError::InvalidInput(format!(
                "unsupported output image extension .{extension}; expected .png, .jpg, or .jpeg"
            ))),
            None => Err(StableDiffusionError::InvalidInput(
                "--out must have an image extension: .png, .jpg, or .jpeg".to_string(),
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StableDiffusionRuntimeOptions {
    pub prompt: String,
    pub negative_prompt: String,
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub scheduler: StableDiffusionScheduler,
    pub backend: StableDiffusionBackend,
    pub model_dir: Option<PathBuf>,
    pub model_id: String,
    pub revision: String,
    pub download: bool,
    pub output_format: StableDiffusionOutputFormat,
    pub stats: bool,
    pub python: String,
}

impl StableDiffusionRuntimeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.prompt.trim().is_empty() {
            return Err(StableDiffusionError::InvalidInput(
                "prompt must not be empty".to_string(),
            ));
        }
        if self.steps == 0 {
            return Err(StableDiffusionError::InvalidInput(
                "steps must be > 0".to_string(),
            ));
        }
        if !self.guidance_scale.is_finite() || self.guidance_scale < 0.0 {
            return Err(StableDiffusionError::InvalidInput(
                "guidance-scale must be finite and >= 0".to_string(),
            ));
        }
        validate_dimensions(self.width, self.height)?;
        StableDiffusionOutputFormat::infer(&self.out)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StableDiffusionGenerationMetadata {
    pub backend: StableDiffusionBackend,
    pub model_source: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub steps: usize,
    pub scheduler: StableDiffusionScheduler,
    pub guidance_scale: f32,
    pub output_path: PathBuf,
    pub elapsed: Option<Duration>,
}

pub fn generate_stable_diffusion(
    options: &StableDiffusionRuntimeOptions,
) -> Result<StableDiffusionGenerationMetadata> {
    options.validate()?;
    match options.backend {
        StableDiffusionBackend::PythonDiffusers => generate_stable_diffusion_python(options),
        StableDiffusionBackend::Rust => generate_stable_diffusion_rust(options),
    }
}

fn generate_stable_diffusion_rust(
    options: &StableDiffusionRuntimeOptions,
) -> Result<StableDiffusionGenerationMetadata> {
    let model_dir = options.model_dir.clone().ok_or_else(|| {
        StableDiffusionError::Asset(
            "native Rust backend requires --model-dir with a Diffusers-format Stable Diffusion 1.x directory".to_string(),
        )
    })?;
    eprintln!(
        "stable-diffusion: preparing native assets in {}",
        model_dir.display()
    );
    let paths = prepare_stable_diffusion_assets(
        &options.model_id,
        &options.revision,
        &model_dir,
        options.download,
    )?;
    eprintln!("stable-diffusion: validating native asset layout");
    validate_stable_diffusion_native_assets(&paths)?;
    eprintln!("stable-diffusion: loading CLIP, UNet, VAE, scheduler, and tokenizer");
    let pipeline = StableDiffusionPipeline::from_assets(paths)?;
    eprintln!("stable-diffusion: starting native generation");
    pipeline.generate(options)
}

pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "width and height must be > 0".to_string(),
        ));
    }
    if width % 8 != 0 || height % 8 != 0 {
        return Err(StableDiffusionError::InvalidInput(
            "width and height must be multiples of 8".to_string(),
        ));
    }
    const MAX_REFERENCE_PIXELS: u32 = 1024 * 1024;
    if width.saturating_mul(height) > MAX_REFERENCE_PIXELS {
        return Err(StableDiffusionError::Unsupported(format!(
            "requested dimensions {width}x{height} exceed the initial reference limit of {MAX_REFERENCE_PIXELS} pixels"
        )));
    }
    Ok(())
}

pub fn model_source_for_display(options: &StableDiffusionRuntimeOptions) -> String {
    options
        .model_dir
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| {
            if options.model_id.is_empty() {
                STABLE_DIFFUSION_V1_5_MODEL_ID.to_string()
            } else {
                options.model_id.clone()
            }
        })
}

pub fn paths_from_options(
    options: &StableDiffusionRuntimeOptions,
) -> Option<StableDiffusionAssetPaths> {
    options
        .model_dir
        .as_ref()
        .map(StableDiffusionAssetPaths::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> StableDiffusionRuntimeOptions {
        StableDiffusionRuntimeOptions {
            prompt: "hello".to_string(),
            negative_prompt: String::new(),
            out: PathBuf::from("/tmp/sd.png"),
            width: 512,
            height: 512,
            steps: 25,
            guidance_scale: 7.5,
            seed: 42,
            scheduler: StableDiffusionScheduler::Ddim,
            backend: StableDiffusionBackend::PythonDiffusers,
            model_dir: None,
            model_id: STABLE_DIFFUSION_V1_5_MODEL_ID.to_string(),
            revision: "main".to_string(),
            download: false,
            output_format: StableDiffusionOutputFormat::Png,
            stats: false,
            python: "python3".to_string(),
        }
    }

    #[test]
    fn validates_runtime_options() {
        let mut options = options();
        assert!(options.validate().is_ok());

        options.prompt = " \t ".to_string();
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("prompt"));
    }

    #[test]
    fn validates_dimensions() {
        assert!(validate_dimensions(512, 512).is_ok());
        assert!(validate_dimensions(0, 512).is_err());
        assert!(validate_dimensions(510, 512).is_err());
    }

    #[test]
    fn infers_output_format() {
        assert_eq!(
            StableDiffusionOutputFormat::infer(PathBuf::from("out.png").as_path()).unwrap(),
            StableDiffusionOutputFormat::Png
        );
        assert_eq!(
            StableDiffusionOutputFormat::infer(PathBuf::from("out.jpeg").as_path()).unwrap(),
            StableDiffusionOutputFormat::Jpeg
        );
        assert!(StableDiffusionOutputFormat::infer(PathBuf::from("out.gif").as_path()).is_err());
    }
}
