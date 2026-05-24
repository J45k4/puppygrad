#!/usr/bin/env python3
"""Generate compact Stable Diffusion parity fixtures.

This script is intentionally outside the Rust runtime. It uses Python
Diffusers/Transformers/Torch as the reference path and writes small JSON files
that Rust tests can compare against without committing model weights or images.
"""

import argparse
import json
import platform
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", choices=["ddim", "ddpm", "euler"], default="ddim")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--include-initial-latents",
        action="store_true",
        help="include full initial latent values so Rust can recompute UNet/VAE parity locally",
    )
    return parser.parse_args()


def tensor_stats(tensor):
    flat = tensor.detach().cpu().float().flatten()
    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "rms": float((flat * flat).mean().sqrt().item()),
    }


def tensor_summary(tensor, slice_len=16, include_values=False):
    flat = tensor.detach().cpu().float().flatten()
    summary = {
        "shape": list(tensor.shape),
        "slice": flat[:slice_len].tolist(),
        "stats": tensor_stats(tensor),
    }
    if include_values:
        summary["values"] = flat.tolist()
    return summary


def main():
    args = parse_args()

    import diffusers
    import torch
    import transformers
    from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline

    scheduler_classes = {
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler,
        "euler": EulerDiscreteScheduler,
    }
    source = args.model_dir or args.model_id
    load_kwargs = {
        "safety_checker": None,
        "feature_extractor": None,
        "image_processor": None,
        "requires_safety_checker": False,
    }
    if args.revision and not args.model_dir:
        load_kwargs["revision"] = args.revision
    if not args.download and not args.model_dir:
        load_kwargs["local_files_only"] = True

    pipe = StableDiffusionPipeline.from_pretrained(source, **load_kwargs)
    pipe.scheduler = scheduler_classes[args.scheduler].from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.steps)

    tokenizer = pipe.tokenizer
    torch.set_grad_enabled(False)
    text_inputs = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_inputs = tokenizer(
        args.negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    prompt_embeddings = pipe.text_encoder(text_inputs.input_ids)[0].detach().cpu().float()
    negative_embeddings = pipe.text_encoder(uncond_inputs.input_ids)[0].detach().cpu().float()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 2 ** (len(pipe.vae.config.block_out_channels) - 1))
    latent_shape = [
        1,
        pipe.unet.config.in_channels,
        args.height // vae_scale_factor,
        args.width // vae_scale_factor,
    ]
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32)
    init_noise_sigma = float(getattr(pipe.scheduler, "init_noise_sigma", 1.0))
    latents = latents * init_noise_sigma

    first_timestep = pipe.scheduler.timesteps[0]
    latent_model_input = pipe.scheduler.scale_model_input(latents, first_timestep)
    noise_uncond = pipe.unet(
        latent_model_input,
        first_timestep,
        encoder_hidden_states=negative_embeddings,
    ).sample.detach().cpu().float()
    noise_cond = pipe.unet(
        latent_model_input,
        first_timestep,
        encoder_hidden_states=prompt_embeddings,
    ).sample.detach().cpu().float()
    guided_noise = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
    first_step_latents = pipe.scheduler.step(
        guided_noise,
        first_timestep,
        latents.detach().cpu().float(),
    ).prev_sample.detach().cpu().float()
    vae_scaling_factor = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
    decoded = pipe.vae.decode(first_step_latents / vae_scaling_factor).sample.detach().cpu().float()
    rgb = (decoded / 2 + 0.5).clamp(0, 1)

    fixture = {
        "metadata": {
            "model_source": source,
            "revision": args.revision,
            "diffusers_version": diffusers.__version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "seed": args.seed,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "scheduler": args.scheduler,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "width": args.width,
            "height": args.height,
        },
        "tokenizer": {
            "conditional_token_ids": text_inputs.input_ids[0].tolist(),
            "unconditional_token_ids": uncond_inputs.input_ids[0].tolist(),
            "conditional_attention_mask": getattr(text_inputs, "attention_mask", torch.ones_like(text_inputs.input_ids))[0].tolist(),
            "unconditional_attention_mask": getattr(uncond_inputs, "attention_mask", torch.ones_like(uncond_inputs.input_ids))[0].tolist(),
        },
        "clip": {
            "conditional_embedding": tensor_summary(prompt_embeddings),
            "unconditional_embedding": tensor_summary(negative_embeddings),
        },
        "scheduler": {
            "timesteps": [int(t) for t in pipe.scheduler.timesteps.detach().cpu().tolist()],
            "init_noise_sigma": init_noise_sigma,
            "alphas_cumprod_head": pipe.scheduler.alphas_cumprod[:8].detach().cpu().float().tolist()
            if hasattr(pipe.scheduler, "alphas_cumprod")
            else [],
            "alphas_cumprod_tail": pipe.scheduler.alphas_cumprod[-8:].detach().cpu().float().tolist()
            if hasattr(pipe.scheduler, "alphas_cumprod")
            else [],
        },
        "latents": {
            "initial": tensor_summary(
                latents,
                include_values=args.include_initial_latents,
            ),
            "first_step": tensor_summary(first_step_latents),
        },
        "unet": {
            "first_timestep": int(first_timestep.item() if hasattr(first_timestep, "item") else first_timestep),
            "latent_model_input": tensor_summary(latent_model_input),
            "unconditional_noise": tensor_summary(noise_uncond),
            "conditional_noise": tensor_summary(noise_cond),
            "guided_noise": tensor_summary(guided_noise),
        },
        "vae": {
            "scaling_factor": vae_scaling_factor,
            "decoded": tensor_summary(decoded),
        },
        "image": {
            "rgb": tensor_summary(rgb),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
