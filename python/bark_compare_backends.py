#!/usr/bin/env python3
"""Run Bark Python and Rust backends and compare WAV-level output properties.

This is local developer tooling for parity validation. It intentionally invokes
the CLI backends as separate commands; the Rust backend itself must not call it.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import tempfile
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("./target/release/puppygrad"))
    parser.add_argument("--python", default="python3")
    parser.add_argument("--model-dir", type=Path, default=Path("models/bark-small"))
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice-preset", default=None)
    parser.add_argument("--seed", type=int, default=299792458)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--semantic-temperature", type=float, default=None)
    parser.add_argument("--coarse-temperature", type=float, default=None)
    parser.add_argument("--fine-temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-semantic-tokens", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--duration-tolerance", type=float, default=0.05)
    parser.add_argument("--rms-tolerance", type=float, default=0.20)
    return parser.parse_args()


def run_backend(args: argparse.Namespace, backend: str, out: Path) -> None:
    cmd = [
        str(args.binary),
        "bark",
        "--model-dir",
        str(args.model_dir),
        "--backend",
        backend,
        "--python",
        args.python,
        "--text",
        args.text,
        "--seed",
        str(args.seed),
        "--out",
        str(out),
    ]
    if args.voice_preset:
        cmd.extend(["--voice-preset", args.voice_preset])
    if args.greedy:
        cmd.append("--greedy")
    for flag, value in [
        ("--semantic-temperature", args.semantic_temperature),
        ("--coarse-temperature", args.coarse_temperature),
        ("--fine-temperature", args.fine_temperature),
        ("--top-k", args.top_k),
        ("--top-p", args.top_p),
        ("--max-semantic-tokens", args.max_semantic_tokens),
        ("--threads", args.threads),
    ]:
        if value is not None:
            cmd.extend([flag, str(value)])
    subprocess.run(cmd, check=True)


def wav_stats(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        sample_width = wav.getsampwidth()
        data = wav.readframes(frames)
    if channels != 1:
        raise RuntimeError(f"{path} has {channels} channels, expected mono")
    if sample_width != 2:
        raise RuntimeError(f"{path} has {sample_width}-byte samples, expected PCM16")

    samples = []
    for idx in range(0, len(data), 2):
        value = int.from_bytes(data[idx : idx + 2], byteorder="little", signed=True)
        samples.append(value / 32768.0)
    if not samples:
        raise RuntimeError(f"{path} contains no samples")
    if not all(math.isfinite(sample) for sample in samples):
        raise RuntimeError(f"{path} contains non-finite samples")

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_count": len(samples),
        "duration": len(samples) / sample_rate,
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "rms": math.sqrt(statistics.fmean(sample * sample for sample in samples)),
    }


def assert_close(left: float, right: float, tolerance: float, label: str) -> None:
    diff = abs(left - right)
    allowed = max(tolerance, abs(right) * tolerance)
    if diff > allowed:
        raise AssertionError(f"{label}: {left} vs {right}, diff {diff}, tolerance {allowed}")


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="puppygrad-bark-compare-") as tmp:
        tmp_path = Path(tmp)
        python_out = tmp_path / "python-transformers.wav"
        rust_out = tmp_path / "rust.wav"
        run_backend(args, "python-transformers", python_out)
        run_backend(args, "rust", rust_out)

        python_stats = wav_stats(python_out)
        rust_stats = wav_stats(rust_out)
        if python_stats["sample_rate"] != rust_stats["sample_rate"]:
            raise AssertionError(f"sample rates differ: {python_stats} vs {rust_stats}")
        if python_stats["channels"] != rust_stats["channels"]:
            raise AssertionError(f"channel counts differ: {python_stats} vs {rust_stats}")
        assert_close(
            float(rust_stats["duration"]),
            float(python_stats["duration"]),
            args.duration_tolerance,
            "duration",
        )
        assert_close(
            float(rust_stats["rms"]),
            float(python_stats["rms"]),
            args.rms_tolerance,
            "rms",
        )

        print("python-transformers", python_stats)
        print("rust", rust_stats)
        if args.keep:
            keep_dir = Path.cwd() / "target" / "bark-compare"
            keep_dir.mkdir(parents=True, exist_ok=True)
            python_keep = keep_dir / python_out.name
            rust_keep = keep_dir / rust_out.name
            python_keep.write_bytes(python_out.read_bytes())
            rust_keep.write_bytes(rust_out.read_bytes())
            print(f"kept {python_keep} and {rust_keep}")


if __name__ == "__main__":
    main()
