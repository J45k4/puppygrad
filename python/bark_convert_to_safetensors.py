#!/usr/bin/env python3
"""Convert a Hugging Face Bark PyTorch checkpoint to safetensors.

This is developer tooling for the Rust backend. The Rust runtime consumes the
resulting model.safetensors file and never imports Python, Torch, or
Transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError(f"unsupported checkpoint type: {type(checkpoint).__name__}")


def main() -> None:
    args = parse_args()
    input_path = args.input or args.model_dir / "pytorch_model.bin"
    output_path = args.output or args.model_dir / "model.safetensors"
    manifest_path = args.manifest or output_path.with_suffix(".manifest.json")

    import torch
    from safetensors.torch import save_file

    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    tensors = {}
    manifest = []
    for name, value in sorted(state_dict.items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous().clone()
        tensors[name] = tensor
        manifest.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
            }
        )

    if not tensors:
        raise RuntimeError(f"no tensors found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {len(tensors)} tensors")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
