//! Stable Diffusion 1.x checkpoint provider. Every model stage executes compiler-generated CPU or GPU kernels.
pub mod stages;
use super::stable_diffusion as sd;
use crate::{
    compiler::{
        cpu::{self, Tensor},
        device::{self, Device},
    },
    runtime::image_ffi::{self as ffi, Api, Callbacks, ErrorBuffer, Info},
};
use serde::Deserialize;
use stages::{Result, Weights};
use std::{
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    time::Instant,
};
#[derive(Deserialize)]
struct Config {
    source: PathBuf,
    model_dir: PathBuf,
    device: String,
    threads: Option<usize>,
    #[serde(default)]
    cpu_target: cpu::CpuTarget,
}
#[derive(Deserialize)]
struct Request {
    prompt: String,
    negative_prompt: String,
    width: u32,
    height: u32,
    steps: usize,
    guidance_scale: f32,
    seed: u64,
}
struct Stage {
    weights: Weights,
    executable: device::Executable,
}
impl Stage {
    fn run(&self, threads: usize) -> Result<Tensor> {
        let start = Instant::now();
        let mut out = self
            .executable
            .run_with_threads(&self.weights.inputs, threads)?;
        if out.len() != 1 {
            return Err("image stage must have one output".into());
        }
        let out = out.remove(0);
        if let Tensor::F32(values) = &out {
            if values.iter().any(|x| !x.is_finite()) {
                return Err("image stage produced non-finite values".into());
            }
        }
        eprintln!("  compiled stage: {:.3}s", start.elapsed().as_secs_f64());
        Ok(out)
    }
}
struct State {
    template: String,
    options: cpu::BuildOptions,
    device: Device,
    threads: usize,
    tokenizer: sd::StableDiffusionTokenizer,
    clip: Stage,
    unet_weights: Weights,
    vae_weights: Weights,
    unet_cfg: sd::Unet2DConditionConfig,
    vae_cfg: sd::AutoencoderKlConfig,
    scheduler: sd::DdimScheduler,
    set_alpha_to_one: bool,
    stages: Option<(u32, u32, Stage, Stage, Stage)>,
    output: Option<(Info, Vec<u8>)>,
}
fn compile_stage(
    program: &crate::compiler::source::Program,
    device: Device,
    options: &cpu::BuildOptions,
    label: &str,
) -> Result<device::Executable> {
    // Stages use independent runtimes because their weight/input slots overlap.
    let executable = device::compile_profiled(program, device, options)?;
    eprintln!(
        "image {label}: {} {} kernels; source: {}",
        executable.kernel_count(),
        device.label(),
        executable.source_path().display()
    );
    Ok(executable)
}
fn read_config<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
impl State {
    fn build(c: Config) -> Result<Self> {
        let device = Device::parse(&c.device)?;
        device.validate_target(c.cpu_target)?;
        let threads = c.threads.unwrap_or_else(cpu::default_threads);
        if threads == 0 {
            return Err("threads must be positive".into());
        }
        c.cpu_target.validate()?;
        let clip_cfg: sd::ClipTextConfig =
            read_config(&c.model_dir.join("text_encoder/config.json"))?;
        sd::config::validate_clip_text_config(&clip_cfg)?;
        let unet_cfg: sd::Unet2DConditionConfig =
            read_config(&c.model_dir.join("unet/config.json"))?;
        let vae_cfg: sd::AutoencoderKlConfig = read_config(&c.model_dir.join("vae/config.json"))?;
        // Tiny public fixtures have Flax class tags but their tensor topology is SD1.
        if !matches!(
            unet_cfg.class_name.as_str(),
            "" | "UNet2DConditionModel" | "FlaxUNet2DConditionModel"
        ) || !matches!(
            vae_cfg.class_name.as_str(),
            "" | "AutoencoderKL" | "FlaxAutoencoderKL"
        ) {
            return Err("unsupported SD1 checkpoint classes".into());
        }
        let mut checked = unet_cfg.clone();
        checked.class_name = "UNet2DConditionModel".into();
        sd::config::validate_unet_config(&checked)?;
        let mut checked = vae_cfg.clone();
        checked.class_name = "AutoencoderKL".into();
        sd::config::validate_vae_config(&checked)?;
        if clip_cfg.max_position_embeddings != 77
            || clip_cfg.hidden_size != unet_cfg.cross_attention_dim
            || vae_cfg.latent_channels != 4
        {
            return Err("incompatible SD1 text/latent dimensions".into());
        }
        let raw: serde_json::Value = read_config(&c.model_dir.join("unet/config.json"))?;
        for (key, expected) in [
            ("center_input_sample", serde_json::json!(false)),
            ("use_linear_projection", serde_json::json!(false)),
            ("flip_sin_to_cos", serde_json::json!(true)),
            ("freq_shift", serde_json::json!(0)),
            ("norm_eps", serde_json::json!(0.00001)),
            ("downsample_padding", serde_json::json!(1)),
            ("mid_block_scale_factor", serde_json::json!(1)),
            ("act_fn", serde_json::json!("silu")),
        ] {
            if let Some(v) = raw.get(key) {
                if *v != expected {
                    return Err(format!("unsupported SD1 configuration {key}={v}").into());
                }
            }
        }
        let mut scheduler_cfg: sd::DdimSchedulerConfig =
            read_config(&c.model_dir.join("scheduler/scheduler_config.json"))?;
        if scheduler_cfg.class_name == "FlaxDDIMScheduler" {
            scheduler_cfg.class_name = "DDIMScheduler".into();
        }
        sd::config::validate_scheduler_config(&scheduler_cfg)?;
        if scheduler_cfg.prediction_type != "epsilon"
            || scheduler_cfg.clip_sample
            || scheduler_cfg
                .timestep_spacing
                .as_deref()
                .unwrap_or("leading")
                != "leading"
        {
            return Err(
                "image provider requires epsilon DDIM with leading timesteps and no clipping"
                    .into(),
            );
        }
        let raw: serde_json::Value =
            read_config(&c.model_dir.join("scheduler/scheduler_config.json"))?;
        let set_alpha_to_one = raw
            .get("set_alpha_to_one")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let scheduler = sd::DdimScheduler::new(scheduler_cfg)?;
        let tokenizer = sd::StableDiffusionTokenizer::from_diffusers_files(
            c.model_dir.join("tokenizer/tokenizer.json"),
            c.model_dir.join("tokenizer/vocab.json"),
            c.model_dir.join("tokenizer/merges.txt"),
        )?;
        let template = std::fs::read_to_string(c.source)?;
        let options = cpu::BuildOptions {
            cpu_target: c.cpu_target,
        };
        eprintln!("loading SD1 checkpoint {}", c.model_dir.display());
        let mut weights =
            Weights::load(&c.model_dir.join("text_encoder/model.safetensors"), false)?;
        let program = stages::clip(&template, &mut weights, &clip_cfg)?;
        let clip = Stage {
            executable: compile_stage(&program, device, &options, "CLIP")?,
            weights,
        };
        eprintln!("loading UNet weights");
        let unet_weights = Weights::load(
            &c.model_dir.join("unet/diffusion_pytorch_model.safetensors"),
            false,
        )?;
        eprintln!("loading VAE decoder weights");
        let vae_weights = Weights::load(
            &c.model_dir.join("vae/diffusion_pytorch_model.safetensors"),
            true,
        )?;
        Ok(Self {
            template,
            options,
            device,
            threads,
            tokenizer,
            clip,
            unet_weights,
            vae_weights,
            unet_cfg,
            vae_cfg,
            scheduler,
            set_alpha_to_one,
            stages: None,
            output: None,
        })
    }
    fn generate(&mut self, r: Request, callbacks: Option<&Callbacks>) -> Result<()> {
        self.output = None;
        let scale = 1u32
            .checked_shl((self.vae_cfg.up_block_types.len() - 1) as u32)
            .ok_or("invalid VAE scale")?;
        let alignment = scale
            .checked_mul(
                1u32.checked_shl((self.unet_cfg.block_out_channels.len() - 1) as u32)
                    .ok_or("invalid UNet scale")?,
            )
            .ok_or("invalid model scale")?;
        if r.width == 0
            || r.height == 0
            || r.width % alignment != 0
            || r.height % alignment != 0
            || r.width as u64 * r.height as u64 > 1024 * 1024
            || !r.guidance_scale.is_finite()
            || r.guidance_scale < 0.0
        {
            return Err(format!("dimensions must be positive multiples of {alignment}, at most 1 megapixel; guidance must be finite and nonnegative").into());
        }
        self.scheduler.set_timesteps(r.steps)?;
        if self
            .scheduler
            .timesteps
            .iter()
            .any(|t| *t >= self.scheduler.alphas_cumprod.len())
        {
            return Err("scheduler timestep offset is out of range".into());
        }
        let tokens = self
            .tokenizer
            .encode_conditioning(&r.prompt, &r.negative_prompt)?;
        let conditionings =
            [tokens.negative_prompt, tokens.prompt].map(|tokens| -> Result<Tensor> {
                self.clip.weights.set(
                    "tokens",
                    Tensor::I32(
                        tokens
                            .token_ids
                            .iter()
                            .map(|x| *x as i32)
                            .collect::<Vec<_>>()
                            .into(),
                    ),
                );
                self.clip.run(self.threads)
            });
        let [uncond, cond] = conditionings;
        let uncond = uncond?;
        let cond = cond?;
        if !self
            .stages
            .as_ref()
            .is_some_and(|s| s.0 == r.width && s.1 == r.height)
        {
            let (mut uw, mut vw) = (self.unet_weights.clone(), self.vae_weights.clone());
            let (h, w) = ((r.height / scale) as usize, (r.width / scale) as usize);
            let p = stages::unet(&self.template, &mut uw, &self.unet_cfg, h, w, 77)?;
            let u = Stage {
                executable: compile_stage(&p, self.device, &self.options, "UNet")?,
                weights: uw,
            };
            let p = stages::vae(&self.template, &mut vw, &self.vae_cfg, h, w, true)?;
            let v = Stage {
                executable: compile_stage(&p, self.device, &self.options, "VAE")?,
                weights: vw,
            };
            let (p, weights) = stages::step(&self.template, h, w)?;
            let step = Stage {
                executable: compile_stage(&p, self.device, &self.options, "DDIM")?,
                weights,
            };
            self.stages = Some((r.width, r.height, u, v, step));
        }
        let (_, _, unet, vae, step) = self.stages.as_mut().unwrap();
        let noise = sd::deterministic_latents_with_scale(r.seed, 1, 4, r.height, r.width, scale)?;
        let mut sample = Tensor::F32(noise.data().to_vec().into());
        let progress = |n| {
            if let Some(c) = callbacks {
                if let Some(f) = c.on_progress {
                    unsafe { f(c.user, n, r.steps as u32) };
                }
            }
        };
        progress(0);
        for (i, &t) in self.scheduler.timesteps.iter().enumerate() {
            eprintln!("denoising {}/{} (t={t})", i + 1, r.steps);
            unet.weights.set("sample", sample.clone());
            unet.weights
                .set("timestep", Tensor::F32(vec![t as f32].into()));
            unet.weights.set("conditioning", uncond.clone());
            let u = unet.run(self.threads)?;
            unet.weights.set("conditioning", cond.clone());
            let c = unet.run(self.threads)?;
            let previous = t as i64 - (self.scheduler.config.num_train_timesteps / r.steps) as i64;
            let alpha = self.scheduler.alphas_cumprod[t];
            let prev = if previous < 0 {
                if self.set_alpha_to_one {
                    1.0
                } else {
                    self.scheduler.alphas_cumprod[0]
                }
            } else {
                self.scheduler.alphas_cumprod[previous as usize]
            };
            step.weights.set("sample", sample);
            step.weights.set("uncond", u);
            step.weights.set("cond", c);
            for (n, v) in [
                ("scale", r.guidance_scale),
                ("alpha", alpha),
                ("previous_alpha", prev),
            ] {
                step.weights.set(n, Tensor::F32(vec![v].into()));
            }
            sample = step.run(self.threads)?;
            progress((i + 1) as u32);
        }
        eprintln!("decoding image");
        vae.weights.set("sample", sample);
        let Tensor::U8(pixels) = vae.run(self.threads)? else {
            return Err("VAE did not return RGB8 pixels".into());
        };
        let info = Info {
            width: r.width,
            height: r.height,
            format: ffi::RGB8,
            reserved: 0,
            stride: r.width as usize * 3,
            bytes: r.width as usize * r.height as usize * 3,
        };
        info.validate().map_err(|e| e.to_string())?;
        if pixels.len() != info.bytes {
            return Err("VAE output size does not match image dimensions".into());
        }
        self.output = Some((info, pixels.to_vec()));
        Ok(())
    }
}
pub static API: Api = Api {
    abi_version: ffi::ABI_VERSION,
    struct_size: std::mem::size_of::<Api>() as u32,
    build_model: Some(build_model),
    infer: Some(infer),
    read_output: Some(read_output),
    free_model: Some(free_model),
};
fn boundary<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err("image provider panicked".into()))
}
unsafe fn error_to(e: *mut ErrorBuffer, message: &str) {
    if let Some(e) = e.as_mut() {
        e.set(message);
    }
}
unsafe extern "C" fn build_model(
    config: *const u8,
    len: usize,
    error: *mut ErrorBuffer,
) -> *mut c_void {
    match boundary(|| {
        if config.is_null() {
            return Err("null image config".into());
        }
        let c = serde_json::from_slice(std::slice::from_raw_parts(config, len))?;
        State::build(c)
    }) {
        Ok(s) => Box::into_raw(Box::new(s)).cast(),
        Err(e) => {
            error_to(error, &e.to_string());
            std::ptr::null_mut()
        }
    }
}
unsafe extern "C" fn infer(
    state: *mut c_void,
    request: *const u8,
    len: usize,
    callbacks: *const Callbacks,
    error: *mut ErrorBuffer,
) -> i32 {
    let status = match boundary(|| {
        let s = state.cast::<State>().as_mut().ok_or("null image state")?;
        s.output = None;
        if request.is_null() {
            return Err("null image request".into());
        }
        s.generate(
            serde_json::from_slice(std::slice::from_raw_parts(request, len))?,
            callbacks.as_ref(),
        )
    }) {
        Ok(()) => 0,
        Err(e) => {
            error_to(error, &e.to_string());
            1
        }
    };
    if let Some(c) = callbacks.as_ref() {
        if let Some(f) = c.on_done {
            f(c.user, status);
        }
    }
    status
}
unsafe extern "C" fn read_output(
    state: *mut c_void,
    dst: *mut u8,
    capacity: usize,
    info: *mut Info,
    error: *mut ErrorBuffer,
) -> i32 {
    match boundary(|| {
        let s = state.cast::<State>().as_ref().ok_or("null image state")?;
        let (metadata, pixels) = s.output.as_ref().ok_or("no completed image")?;
        let info = info.as_mut().ok_or("null image metadata")?;
        *info = *metadata;
        if dst.is_null() {
            if capacity != 0 {
                return Err("null image output with nonzero capacity".into());
            }
        } else {
            if capacity < pixels.len() {
                return Err("image output buffer too small".into());
            }
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), dst, pixels.len());
        }
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            error_to(error, &e.to_string());
            1
        }
    }
}
unsafe extern "C" fn free_model(state: *mut c_void) {
    if !state.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(state.cast::<State>()))
        }));
    }
}

/// Fetch fp16 SD1.5 files into the canonical Diffusers paths used by the loader.
pub fn download_sd15(dir: &std::path::Path) -> Result<()> {
    const MODEL: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
    const REVISION: &str = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14";
    for filename in sd::STABLE_DIFFUSION_NATIVE_REQUIRED_FILES {
        let remote = match filename {
            "text_encoder/model.safetensors" => "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors" => {
                "unet/diffusion_pytorch_model.fp16.safetensors"
            }
            "vae/diffusion_pytorch_model.safetensors" => {
                "vae/diffusion_pytorch_model.fp16.safetensors"
            }
            other => other,
        };
        crate::models::assets::download_huggingface_file(
            MODEL,
            REVISION,
            remote,
            &dir.join(filename),
        )?;
    }
    Ok(())
}
