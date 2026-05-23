#!/usr/bin/env python3
"""Generate Python Transformers Bark reference fixtures.

The output is intentionally a compact JSON fixture for Rust parity tests. This
script is developer tooling only; the Rust backend must not call it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--voice-preset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=299792458)
    parser.add_argument("--semantic-temperature", type=float, default=None)
    parser.add_argument("--coarse-temperature", type=float, default=None)
    parser.add_argument("--fine-temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-semantic-tokens", type=int, default=None)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force semantic/coarse/fine temperatures to 0 unless explicitly overridden.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waveform-samples", type=int, default=256)
    return parser.parse_args()


def version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def waveform_stats(samples: Any) -> dict[str, float | int | list[float]]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise RuntimeError("Transformers generated an empty waveform")
    return {
        "sample_count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
    }


def tensor_shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def int_tensor(value: Any) -> Any:
    return value.detach().cpu().numpy().astype(int).tolist()


def float_tensor(value: Any) -> Any:
    return value.detach().cpu().numpy().astype(float).tolist()


def topk_summary(logits: Any, k: int = 10) -> dict[str, Any]:
    import torch

    finite_logits = logits.detach().float()
    values, indices = torch.topk(finite_logits, min(k, finite_logits.shape[-1]), dim=-1)
    probabilities = torch.softmax(finite_logits, dim=-1).gather(-1, indices)
    return {
        "token_ids": indices.detach().cpu().numpy().astype(int).tolist(),
        "logits": values.detach().cpu().numpy().astype(float).tolist(),
        "probabilities": probabilities.detach().cpu().numpy().astype(float).tolist(),
    }


def first_batch_flat_int(value: Any) -> list[int]:
    return value.detach().cpu().reshape(value.shape[0], -1)[0].numpy().astype(int).tolist()


def token_strings(processor: Any, input_ids: Any) -> list[str]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "convert_ids_to_tokens"):
        return []
    return [
        str(token)
        for token in tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    ]


def normalized_text(processor: Any, text: str) -> str | None:
    tokenizer = getattr(processor, "tokenizer", None)
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend_tokenizer, "normalizer", None)
    if normalizer is not None and hasattr(normalizer, "normalize_str"):
        return str(normalizer.normalize_str(text))
    return None


def split_generation_kwargs(
    args: argparse.Namespace, attention_mask: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common = {}
    if args.top_k is not None:
        common["top_k"] = args.top_k
    if args.top_p is not None:
        common["top_p"] = args.top_p

    semantic_kwargs = dict(common)
    semantic_kwargs["attention_mask"] = attention_mask
    if args.greedy:
        semantic_kwargs["do_sample"] = False
    if args.semantic_temperature is not None:
        semantic_kwargs["temperature"] = args.semantic_temperature
    if args.max_semantic_tokens is not None:
        semantic_kwargs["max_new_tokens"] = args.max_semantic_tokens

    coarse_kwargs = dict(common)
    if args.greedy:
        coarse_kwargs["do_sample"] = False
    if args.coarse_temperature is not None:
        coarse_kwargs["temperature"] = args.coarse_temperature

    fine_kwargs = dict(common)
    if args.greedy:
        fine_kwargs["temperature"] = 1.0
    if args.fine_temperature is not None:
        fine_kwargs["temperature"] = args.fine_temperature

    return semantic_kwargs, coarse_kwargs, fine_kwargs


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    import transformers
    from transformers import AutoProcessor, BarkModel
    try:
        from transformers import (
            BarkCoarseGenerationConfig,
            BarkFineGenerationConfig,
            BarkSemanticGenerationConfig,
        )
    except ImportError:
        from transformers.models.bark.modeling_bark import (
            BarkCoarseGenerationConfig,
            BarkFineGenerationConfig,
            BarkSemanticGenerationConfig,
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = BarkModel.from_pretrained(args.model_dir)
    model.eval()

    processor_kwargs = {}
    if args.voice_preset:
        processor_kwargs["voice_preset"] = args.voice_preset
    inputs = processor(args.text, **processor_kwargs)
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    history_prompt = model_inputs.get("history_prompt")
    semantic_generation_config = BarkSemanticGenerationConfig(
        **model.generation_config.semantic_config
    )
    coarse_generation_config = BarkCoarseGenerationConfig(
        **model.generation_config.coarse_acoustics_config
    )
    fine_generation_config = BarkFineGenerationConfig(
        **model.generation_config.fine_acoustics_config
    )
    codebook_size = int(model.generation_config.codebook_size)
    semantic_kwargs, coarse_kwargs, fine_kwargs = split_generation_kwargs(args, attention_mask)

    with torch.no_grad():
        semantic_model_input = input_ids.clone() + semantic_generation_config.text_encoding_offset
        if attention_mask is not None:
            semantic_model_input = semantic_model_input.masked_fill(
                (1 - attention_mask).bool(), semantic_generation_config.text_pad_token
            )
        max_semantic_len = semantic_generation_config.max_input_semantic_length
        if history_prompt is not None:
            semantic_history = history_prompt["semantic_prompt"][-max_semantic_len:]
            semantic_history = torch.nn.functional.pad(
                semantic_history,
                (0, max_semantic_len - len(semantic_history)),
                value=semantic_generation_config.semantic_pad_token,
                mode="constant",
            )
        else:
            semantic_history = torch.full(
                (max_semantic_len,),
                semantic_generation_config.semantic_pad_token,
                device=model.device,
                dtype=torch.int,
            )
        semantic_history = torch.repeat_interleave(
            semantic_history[None], semantic_model_input.shape[0], dim=0
        )
        semantic_infer = torch.tensor(
            [[semantic_generation_config.semantic_infer_token]] * semantic_model_input.shape[0],
            dtype=torch.int,
            device=model.device,
        )
        semantic_inputs_embeds = torch.cat(
            [
                model.semantic.input_embeds_layer(semantic_model_input[:, :max_semantic_len])
                + model.semantic.input_embeds_layer(semantic_history[:, : max_semantic_len + 1]),
                model.semantic.input_embeds_layer(semantic_infer),
            ],
            dim=1,
        )
        semantic_first_logits_tensor = model.semantic(
            inputs_embeds=semantic_inputs_embeds,
            use_cache=False,
        ).logits[:, -1, :]
        semantic_first_logits_tensor[
            :, semantic_generation_config.semantic_vocab_size : semantic_generation_config.semantic_pad_token
        ] = -float("inf")
        semantic_first_logits_tensor[:, semantic_generation_config.semantic_pad_token + 1 :] = -float(
            "inf"
        )

        semantic_output = model.semantic.generate(
            input_ids,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            **semantic_kwargs,
        )
        coarse_output, coarse_output_lengths = model.coarse_acoustics.generate(
            semantic_output.clone(),
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            codebook_size=codebook_size,
            return_output_lengths=True,
            **coarse_kwargs,
        )
        semantic_for_coarse = semantic_output.clone()
        semantic_for_coarse.masked_fill_(
            semantic_for_coarse == semantic_generation_config.semantic_pad_token,
            coarse_generation_config.coarse_semantic_pad_token,
        )
        semantic_to_coarse_ratio = (
            coarse_generation_config.coarse_rate_hz
            / semantic_generation_config.semantic_rate_hz
            * coarse_generation_config.n_coarse_codebooks
        )
        max_semantic_history = int(
            np.floor(coarse_generation_config.max_coarse_history / semantic_to_coarse_ratio)
        )
        semantic_idx = 0
        coarse_first_input = semantic_for_coarse[
            :, np.max([0, semantic_idx - max_semantic_history]) :
        ]
        coarse_first_input = coarse_first_input[
            :, : coarse_generation_config.max_coarse_input_length
        ]
        coarse_first_input = torch.nn.functional.pad(
            coarse_first_input,
            (0, coarse_generation_config.max_coarse_input_length - coarse_first_input.shape[-1]),
            "constant",
            coarse_generation_config.coarse_semantic_pad_token,
        )
        coarse_first_input = torch.hstack(
            [
                coarse_first_input,
                torch.tensor(
                    [[coarse_generation_config.coarse_infer_token]] * semantic_output.shape[0],
                    device=model.device,
                ),
            ]
        )
        coarse_first_logits_tensor = model.coarse_acoustics(
            coarse_first_input,
            use_cache=False,
        ).logits[:, -1, :]
        first_codebook_start = semantic_generation_config.semantic_vocab_size
        first_codebook_end = first_codebook_start + codebook_size
        coarse_first_logits_tensor[:, :first_codebook_start] = -float("inf")
        coarse_first_logits_tensor[:, first_codebook_end:] = -float("inf")

        n_coarse = int(coarse_generation_config.n_coarse_codebooks)
        initial_fine_input = coarse_output.view(coarse_output.shape[0], -1, n_coarse)
        initial_fine_input = torch.remainder(
            initial_fine_input - semantic_generation_config.semantic_vocab_size, codebook_size
        )
        initial_fine_input = torch.nn.functional.pad(
            initial_fine_input,
            (0, fine_generation_config.n_fine_codebooks - n_coarse),
            "constant",
            codebook_size,
        )
        fine_output = model.fine_acoustics.generate(
            coarse_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            fine_generation_config=fine_generation_config,
            codebook_size=codebook_size,
            **fine_kwargs,
        )
        audio = model.codec_decode(fine_output)

    samples = audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
    sample_rate = int(getattr(model.generation_config, "sample_rate", 24000))
    input_ids_list = first_batch_flat_int(input_ids)
    attention_mask_list = first_batch_flat_int(attention_mask) if attention_mask is not None else []
    semantic_tokens = first_batch_flat_int(semantic_output)
    try:
        semantic_eos_position = semantic_tokens.index(semantic_generation_config.eos_token_id)
    except ValueError:
        semantic_eos_position = None
    coarse_codebooks = torch.remainder(
        coarse_output - semantic_generation_config.semantic_vocab_size, codebook_size
    )
    coarse_codebooks = coarse_codebooks.view(
        coarse_codebooks.shape[0], -1, coarse_generation_config.n_coarse_codebooks
    ).transpose(1, 2)

    generation_settings = {}
    if args.greedy:
        generation_settings["greedy"] = True
    if args.semantic_temperature is not None:
        generation_settings["semantic_temperature"] = args.semantic_temperature
    if args.coarse_temperature is not None:
        generation_settings["coarse_temperature"] = args.coarse_temperature
    if args.fine_temperature is not None:
        generation_settings["fine_temperature"] = args.fine_temperature
    if args.top_k is not None:
        generation_settings["top_k"] = args.top_k
    if args.top_p is not None:
        generation_settings["top_p"] = args.top_p
    if args.max_semantic_tokens is not None:
        generation_settings["max_semantic_tokens"] = args.max_semantic_tokens

    with torch.no_grad():
        first_fine_logits = None
        n_coarse = int(coarse_generation_config.n_coarse_codebooks)
        if fine_output.shape[1] > n_coarse:
            fine_input = fine_output.transpose(1, 2)
            logits = model.fine_acoustics.forward(n_coarse, fine_input).logits
            first_fine_logits = {
                "codebook": n_coarse,
                "shape": tensor_shape(logits),
                "first_frame_slice": float_tensor(logits[0, 0, : min(32, codebook_size)]),
            }

        quantized_shape = tensor_shape(
            model.codec_model.quantizer.decode(fine_output.transpose(0, 1))
        )

    fixture = {
        "metadata": {
            "model_dir": args.model_dir,
            "revision": getattr(model.config, "_commit_hash", None),
            "transformers_version": version(transformers),
            "torch_version": version(torch),
            "numpy_version": version(np),
            "seed": args.seed,
            "text": args.text,
            "voice_preset": args.voice_preset,
            "generation_settings": generation_settings,
            "sample_rate": sample_rate,
            "codebook_size": codebook_size,
        },
        "tokenizer": {
            "normalized_text": normalized_text(processor, args.text),
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "tokens": token_strings(processor, input_ids),
        },
        "semantic": {
            "model_input_ids_after_text_offset": first_batch_flat_int(semantic_model_input),
            "generated_tokens": semantic_tokens,
            "eos_position": semantic_eos_position,
            "first_logits": {
                "shape": tensor_shape(semantic_first_logits_tensor),
                "slice": float_tensor(semantic_first_logits_tensor[0, :32]),
                "top_k": topk_summary(semantic_first_logits_tensor[0], 10),
            },
            "shape": tensor_shape(semantic_output),
        },
        "coarse": {
            "first_window_input_ids": first_batch_flat_int(coarse_first_input),
            "first_logits": {
                "shape": tensor_shape(coarse_first_logits_tensor),
                "slice": float_tensor(
                    coarse_first_logits_tensor[
                        0, first_codebook_start : min(first_codebook_end, first_codebook_start + 32)
                    ]
                ),
                "top_k": topk_summary(coarse_first_logits_tensor[0], 10),
            },
            "generated_tokens_flat": first_batch_flat_int(coarse_output),
            "generated_codebooks": int_tensor(coarse_codebooks[0]),
            "output_lengths": first_batch_flat_int(coarse_output_lengths),
            "shape": tensor_shape(coarse_output),
            "codebook_shape": tensor_shape(coarse_codebooks),
        },
        "fine": {
            "input_code_matrix": int_tensor(initial_fine_input[0].transpose(0, 1)),
            "generated_codebooks": int_tensor(fine_output[0]),
            "shape": tensor_shape(fine_output),
            "first_logits": first_fine_logits,
        },
        "decoder": {
            "quantized_latent_shape": quantized_shape,
            **waveform_stats(samples),
            "first_samples": samples[: args.waveform_samples].astype(float).tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
