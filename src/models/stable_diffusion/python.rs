use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::{
    model_source_for_display, Result, StableDiffusionBackend, StableDiffusionError,
    StableDiffusionGenerationMetadata, StableDiffusionRuntimeOptions,
};

#[derive(Debug, Deserialize)]
struct PythonStableDiffusionMetadata {
    backend: String,
    model_source: String,
    width: u32,
    height: u32,
    seed: u64,
    steps: usize,
    scheduler: String,
    guidance_scale: f32,
    elapsed_seconds: f64,
}

pub fn generate_stable_diffusion_python(
    options: &StableDiffusionRuntimeOptions,
) -> Result<StableDiffusionGenerationMetadata> {
    if let Some(parent) = options.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let meta_path = temporary_sidecar_path(&options.out, "json");
    let model_dir_arg = options
        .model_dir
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let start = Instant::now();
    let output = Command::new(&options.python)
        .arg("-c")
        .arg(STABLE_DIFFUSION_PYTHON_DIFFUSERS_SCRIPT)
        .arg(&model_dir_arg)
        .arg(&options.model_id)
        .arg(&options.revision)
        .arg(if options.download { "1" } else { "0" })
        .arg(&options.prompt)
        .arg(&options.negative_prompt)
        .arg(&options.out)
        .arg(&meta_path)
        .arg(options.width.to_string())
        .arg(options.height.to_string())
        .arg(options.steps.to_string())
        .arg(options.guidance_scale.to_string())
        .arg(options.seed.to_string())
        .arg(options.scheduler.label())
        .output()
        .map_err(|err| {
            StableDiffusionError::Python(format!("failed to start {}: {err}", options.python))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = fs::remove_file(&meta_path);
        return Err(StableDiffusionError::Python(format!(
            "python-diffusers backend failed with status {}; install with `python3 -m pip install diffusers torch transformers safetensors Pillow`.\nstderr:\n{}\nstdout:\n{}",
            output.status,
            stderr.trim(),
            stdout.trim()
        )));
    }

    let metadata: PythonStableDiffusionMetadata =
        serde_json::from_str(&fs::read_to_string(&meta_path).map_err(|err| {
            StableDiffusionError::Python(format!(
                "python-diffusers backend did not write metadata {}: {err}",
                meta_path.display()
            ))
        })?)
        .map_err(|err| {
            StableDiffusionError::Python(format!(
                "failed to parse python-diffusers metadata {}: {err}",
                meta_path.display()
            ))
        })?;
    let _ = fs::remove_file(&meta_path);

    let image_metadata = fs::metadata(&options.out).map_err(|err| {
        StableDiffusionError::Image(format!(
            "python-diffusers backend did not write {}: {err}",
            options.out.display()
        ))
    })?;
    if image_metadata.len() == 0 {
        return Err(StableDiffusionError::Image(format!(
            "python-diffusers backend wrote empty image {}",
            options.out.display()
        )));
    }
    if metadata.width != options.width || metadata.height != options.height {
        return Err(StableDiffusionError::Image(format!(
            "python-diffusers backend reported dimensions {}x{}, expected {}x{}",
            metadata.width, metadata.height, options.width, options.height
        )));
    }
    if metadata.backend != "python-diffusers" {
        return Err(StableDiffusionError::Python(format!(
            "unexpected python backend metadata value {}",
            metadata.backend
        )));
    }
    if metadata.scheduler != options.scheduler.label() {
        return Err(StableDiffusionError::Python(format!(
            "python-diffusers backend reported scheduler {}, expected {}",
            metadata.scheduler,
            options.scheduler.label()
        )));
    }

    Ok(StableDiffusionGenerationMetadata {
        backend: StableDiffusionBackend::PythonDiffusers,
        model_source: if metadata.model_source.is_empty() {
            model_source_for_display(options)
        } else {
            metadata.model_source
        },
        width: metadata.width,
        height: metadata.height,
        seed: metadata.seed,
        steps: metadata.steps,
        scheduler: options.scheduler,
        guidance_scale: metadata.guidance_scale,
        output_path: options.out.clone(),
        elapsed: Some(Duration::from_secs_f64(
            metadata
                .elapsed_seconds
                .max(0.0)
                .max(start.elapsed().as_secs_f64()),
        )),
    })
}

fn temporary_sidecar_path(out: &Path, extension: &str) -> std::path::PathBuf {
    let file_name = out
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "stable-diffusion-out".into());
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    out.with_file_name(format!(".{file_name}.{pid}.{nanos}.{extension}"))
}

const STABLE_DIFFUSION_PYTHON_DIFFUSERS_SCRIPT: &str = r#"
import json
import os
import sys
import time

(
    model_dir,
    model_id,
    revision,
    download_text,
    prompt,
    negative_prompt,
    out_path,
    meta_path,
    width_text,
    height_text,
    steps_text,
    guidance_scale_text,
    seed_text,
    scheduler_name,
) = sys.argv[1:15]

download = download_text == "1"
width = int(width_text)
height = int(height_text)
steps = int(steps_text)
guidance_scale = float(guidance_scale_text)
seed = int(seed_text)

try:
    import torch
    from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline
except ModuleNotFoundError as exc:
    print(
        "Missing Python package for python-diffusers backend. "
        "Install with: python3 -m pip install diffusers torch transformers safetensors Pillow",
        file=sys.stderr,
    )
    raise

schedulers = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "euler": EulerDiscreteScheduler,
}
if scheduler_name not in schedulers:
    raise ValueError(f"unsupported scheduler {scheduler_name!r}")

source = model_dir if model_dir else model_id
if not source:
    raise ValueError("model_dir or model_id is required")
if model_dir and not os.path.isdir(model_dir):
    raise FileNotFoundError(f"model directory does not exist: {model_dir}")

load_kwargs = {
    "safety_checker": None,
    "feature_extractor": None,
    "image_processor": None,
    "requires_safety_checker": False,
}
if revision and not model_dir:
    load_kwargs["revision"] = revision
if not download and not model_dir:
    load_kwargs["local_files_only"] = True

start = time.perf_counter()
pipe = StableDiffusionPipeline.from_pretrained(source, **load_kwargs)
pipe.scheduler = schedulers[scheduler_name].from_config(pipe.scheduler.config)
generator = torch.Generator(device=pipe.device).manual_seed(seed)
with torch.no_grad():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

parent = os.path.dirname(out_path)
if parent:
    os.makedirs(parent, exist_ok=True)
image.save(out_path)
elapsed = time.perf_counter() - start
with open(meta_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "backend": "python-diffusers",
            "model_source": source,
            "width": int(image.width),
            "height": int(image.height),
            "seed": seed,
            "steps": steps,
            "scheduler": scheduler_name,
            "guidance_scale": guidance_scale,
            "elapsed_seconds": elapsed,
        },
        handle,
    )
"#;
