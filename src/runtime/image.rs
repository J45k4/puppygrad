//! Generic image host: UTF-8 JSON request in, caller-owned RGB8 buffer out.
use crate::{compiler::cpu::CpuTarget, models::pup_image, runtime::image_ffi};
use std::{path::PathBuf, time::Instant};
#[derive(clap::Args, Debug)]
pub struct Options {
    /// .pup stage source or a shared library exporting get_image_api.
    pub source: PathBuf,
    #[arg(long, default_value = "models/stable-diffusion-v1-5")]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub prompt: String,
    /// Download missing SD1.5 assets from the pinned public checkpoint.
    #[arg(long)]
    pub download: bool,
    #[arg(long, default_value = "")]
    pub negative_prompt: String,
    #[arg(long, default_value = "cpu")]
    pub device: String,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long,value_enum,default_value_t=CpuTarget::Generic)]
    pub cpu_target: CpuTarget,
    #[arg(long, default_value_t = 512)]
    pub width: u32,
    #[arg(long, default_value_t = 512)]
    pub height: u32,
    #[arg(long, default_value_t = 20)]
    pub steps: usize,
    #[arg(long, default_value_t = 7.5)]
    pub guidance_scale: f32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, alias = "out", default_value = "image.png")]
    pub output: PathBuf,
}
pub fn run(o: Options) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let is_pup = o.source.extension().and_then(|s| s.to_str()) == Some("pup");
    if o.width == 0
        || o.height == 0
        || o.steps == 0
        || !o.guidance_scale.is_finite()
        || o.guidance_scale < 0.0
    {
        return Err(
            "image dimensions and steps must be positive; guidance must be finite and nonnegative"
                .into(),
        );
    }
    if is_pup {
        if !matches!(o.device.as_str(), "cpu" | "cpu:0" | "c" | "c:0") {
            return Err("image .pup currently supports --device cpu (generated C)".into());
        }
        if o.threads == Some(0) {
            return Err("threads must be positive".into());
        }
        o.cpu_target.validate()?;
    }
    if o.download {
        if !is_pup {
            return Err("--download applies to the built-in .pup SD1 provider".into());
        }
        crate::models::pup_image::download_sd15(&o.model_dir)?;
    }
    let config = serde_json::to_vec(
        &serde_json::json!({"source":o.source,"model_dir":o.model_dir,"device":o.device,"threads":o.threads,"cpu_target":o.cpu_target}),
    )?;
    let request = serde_json::to_vec(
        &serde_json::json!({"prompt":o.prompt,"negative_prompt":o.negative_prompt,"width":o.width,"height":o.height,"steps":o.steps,"guidance_scale":o.guidance_scale,"seed":o.seed}),
    )?;
    let mut model = unsafe {
        if is_pup {
            image_ffi::Model::from_api(pup_image::API, &config)
        } else {
            if o.cpu_target != CpuTarget::Generic {
                return Err("--cpu-target applies to .pup source compilation".into());
            }
            image_ffi::Model::load(&o.source, &config)
        }
    }
    .map_err(|e| e.to_string())?;
    let out = model.infer(&request, None).map_err(|e| e.to_string())?;
    let image = image::RgbImage::from_raw(out.info.width, out.info.height, out.pixels)
        .ok_or("invalid image output")?;
    image.save(&o.output)?;
    eprintln!(
        "saved {} ({}×{}, {:.2}s including load/compile)",
        o.output.display(),
        out.info.width,
        out.info.height,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
