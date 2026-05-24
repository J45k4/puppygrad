#!/usr/bin/env python3
"""Run Python and native Rust Stable Diffusion backends and summarize outputs."""

import argparse
import json
import struct
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--puppygrad", default="./target/debug/puppygrad")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "ddpm", "euler"])
    parser.add_argument("--python", default="python3")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def image_info(path):
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", data[16:24])
        return {"format": "png", "width": width, "height": height, "bytes": len(data)}
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            while index < len(data) and data[index] == 0xFF:
                index += 1
            marker = data[index]
            index += 1
            if marker in {0xD8, 0xD9}:
                continue
            length = struct.unpack(">H", data[index:index + 2])[0]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3}:
                height, width = struct.unpack(">HH", data[index + 3:index + 7])
                return {"format": "jpeg", "width": width, "height": height, "bytes": len(data)}
            index += length
    raise ValueError(f"unsupported or invalid image file: {path}")


def run_backend(args, backend, out_path):
    command = [
        args.puppygrad,
        "stable-diffusion",
        "--backend",
        backend,
        "--model-dir",
        args.model_dir,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--out",
        str(out_path),
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--scheduler",
        args.scheduler,
    ]
    if backend == "python-diffusers":
        command.extend(["--python", args.python])
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.time() - started
    result = {
        "backend": backend,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "output_path": str(out_path),
    }
    if completed.returncode == 0:
        result["image"] = image_info(out_path)
    return result


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else out_dir / "summary.json"
    results = [
        run_backend(args, "python-diffusers", out_dir / "python.png"),
        run_backend(args, "rust", out_dir / "rust.png"),
    ]
    summary = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "model_dir": args.model_dir,
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "scheduler": args.scheduler,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
