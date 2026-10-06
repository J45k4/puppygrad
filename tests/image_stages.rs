//! Optional checkpoint parity: download the public tiny fixture first (see image-runtime.md).
use puppygrad::{
    compiler::{
        cpu::{self, Tensor},
        device::{self, Device},
    },
    models::{
        pup_image::stages::{self, Weights},
        stable_diffusion as sd,
    },
};
use std::path::Path;
fn read<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> T {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn compare(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{label}: max absolute error {error}");
    assert!(error < tol, "{label}: {error} >= {tol}");
}
#[test]
#[ignore = "requires models/stable-diffusion-tiny checkpoint"]
fn compiled_sd1_stages_match_native_reference() {
    stage_parity(Device::Cpu);
}
#[test]
#[ignore = "requires AMD GPU, HIPRTC and models/stable-diffusion-tiny checkpoint"]
fn hip_sd1_stages_match_native_reference() {
    stage_parity(Device::parse("hip:0").unwrap());
}
fn stage_parity(device: Device) {
    let dir = Path::new("models/stable-diffusion-tiny");
    let template = std::fs::read_to_string("examples/image.pup").unwrap();
    let options = cpu::BuildOptions {
        cpu_target: if device == Device::Cpu {
            cpu::CpuTarget::Native
        } else {
            cpu::CpuTarget::Generic
        },
    };
    let cfg: sd::ClipTextConfig = read(dir.join("text_encoder/config.json"));
    let tokenizer = sd::StableDiffusionTokenizer::from_diffusers_files(
        dir.join("tokenizer/tokenizer.json"),
        dir.join("tokenizer/vocab.json"),
        dir.join("tokenizer/merges.txt"),
    )
    .unwrap();
    let tokens = tokenizer
        .encode_prompt("a puppy in a garden")
        .unwrap()
        .token_ids;
    let encoder = sd::ClipTextEncoder::new(
        cfg.clone(),
        sd::load_clip_text_weights(dir.join("text_encoder/model.safetensors"), &cfg).unwrap(),
    )
    .unwrap();
    let expected = encoder.encode_token_ids(&tokens).unwrap();
    let mut cw = Weights::load(&dir.join("text_encoder/model.safetensors"), false).unwrap();
    let p = stages::clip(&template, &mut cw, &cfg).unwrap();
    let exe = device::compile_profiled(&p, device, &options).unwrap();
    cw.set(
        "tokens",
        Tensor::I32(tokens.iter().map(|x| *x as i32).collect::<Vec<_>>().into()),
    );
    let context = exe.run_with_threads(&cw.inputs, 3).unwrap().remove(0);
    compare("CLIP", context.f32().unwrap(), expected.data(), 2e-4);
    let cfg: sd::Unet2DConditionConfig = read(dir.join("unet/config.json"));
    let weights = sd::load_unet_2d_condition_model_weights(
        dir.join("unet/diffusion_pytorch_model.safetensors"),
        &cfg,
    )
    .unwrap();
    let sample = sd::deterministic_normal_tensor([1, 4, 8, 8], 42).unwrap();
    let mut uw =
        Weights::load(&dir.join("unet/diffusion_pytorch_model.safetensors"), false).unwrap();
    let p = stages::unet(&template, &mut uw, &cfg, 8, 8, 77).unwrap();
    let exe = device::compile_profiled(&p, device, &options).unwrap();
    for timestep in [1, 501] {
        let expected = sd::unet_forward(&sample, timestep, &expected, &weights).unwrap();
        uw.set("sample", Tensor::F32(sample.data().to_vec().into()));
        uw.set("conditioning", context.clone());
        uw.set("timestep", Tensor::F32(vec![timestep as f32].into()));
        let out = exe.run_with_threads(&uw.inputs, 3).unwrap();
        compare(
            &format!("UNet t={timestep}"),
            out[0].f32().unwrap(),
            expected.data(),
            3e-4,
        );
    }
    let cfg: sd::AutoencoderKlConfig = read(dir.join("vae/config.json"));
    let weights = sd::load_vae_decoder_model_weights(
        dir.join("vae/diffusion_pytorch_model.safetensors"),
        &cfg,
    )
    .unwrap();
    let expected = sd::vae_decode_latents(&sample, &cfg, &weights).unwrap();
    let mut vw = Weights::load(&dir.join("vae/diffusion_pytorch_model.safetensors"), true).unwrap();
    let p = stages::vae(&template, &mut vw, &cfg, 8, 8, false).unwrap();
    let exe = device::compile_profiled(&p, device, &options).unwrap();
    vw.set("sample", Tensor::F32(sample.data().to_vec().into()));
    let out = exe.run_with_threads(&vw.inputs, 3).unwrap();
    compare("VAE", out[0].f32().unwrap(), expected.data(), 3e-4);
}

#[test]
#[ignore = "requires models/stable-diffusion-tiny checkpoint"]
fn image_provider_repeats_requests_and_recovers_after_invalid_input() {
    image_provider("cpu", "native");
}
#[test]
#[ignore = "requires AMD GPU, HIPRTC and models/stable-diffusion-tiny checkpoint"]
fn hip_image_provider_repeats_requests_and_recovers_after_invalid_input() {
    image_provider("hip:0", "generic");
}
fn image_provider(device: &str, target: &str) {
    use puppygrad::{models::pup_image, runtime::image_ffi::Model};
    let config=serde_json::to_vec(&serde_json::json!({"source":"examples/image.pup","model_dir":"models/stable-diffusion-tiny","device":device,"threads":3,"cpu_target":target})).unwrap();
    let mut model = unsafe { Model::from_api(pup_image::API, &config).unwrap() };
    let mut request = serde_json::json!({"prompt":"a puppy in a garden","negative_prompt":"","width":16,"height":16,"steps":2,"guidance_scale":7.5,"seed":42});
    let a = model
        .infer(&serde_json::to_vec(&request).unwrap(), None)
        .unwrap();
    let b = model
        .infer(&serde_json::to_vec(&request).unwrap(), None)
        .unwrap();
    assert_eq!(a.pixels, b.pixels);
    request["width"] = serde_json::json!(17);
    assert!(model
        .infer(&serde_json::to_vec(&request).unwrap(), None)
        .is_err());
    request["width"] = serde_json::json!(16);
    request["height"] = serde_json::json!(8);
    let c = model
        .infer(&serde_json::to_vec(&request).unwrap(), None)
        .unwrap();
    assert_eq!(c.pixels.len(), 16 * 8 * 3);
    request["height"] = serde_json::json!(16);
    let d = model
        .infer(&serde_json::to_vec(&request).unwrap(), None)
        .unwrap();
    assert_eq!(a.pixels, d.pixels);
}
