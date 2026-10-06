"""Compare CUDA scheduling modes through actual Qwen3 Model.infer."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess


def stats(values):
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/qwen3-0.6b"))
    parser.add_argument("--source", type=Path, default=Path("examples/qwen3_cached.pup"))
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/qwen3-replay"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--compare", choices=["replay", "reductions", "row-fusion", "expressions"], default="replay")
    args = parser.parse_args()
    if args.warmups < 3 or args.runs < 1 or not 2 <= args.tokens <= 128:
        parser.error("require >=3 warmups, >=1 run, and 2..128 output tokens")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PUPPYGRAD_NVRTC", str(Path(".cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so.12").resolve()))
    deps = Path("target/release/deps")
    runner = out / "runner"
    command = ["rustc", "--edition=2021", "-O", str(Path(__file__).with_name("runner.rs")), "-L", f"dependency={deps}", "-o", str(runner)]
    for name in ["puppygrad", "serde_json", "tokenizers"]:
        library = max(deps.glob(f"lib{name}-*.rlib"), key=lambda p: p.stat().st_mtime)
        command += ["--extern", f"{name}={library}"]
    subprocess.run(command, check=True)
    workers = []
    names = ["direct", "replay"] if args.compare == "replay" else ["baseline", "optimized"]

    def start(mode=None, enabled=True):
        env = dict(os.environ, PUPPYGRAD_CUDA_GRAPH="1" if enabled or args.compare != "replay" else "0")
        if args.compare == "reductions":
            env.update(PUPPYGRAD_CUDA_REDUCTIONS="1" if enabled else "0", PUPPYGRAD_CUDA_ROW_FUSION="1")
        elif args.compare == "row-fusion":
            env.update(PUPPYGRAD_CUDA_REDUCTIONS="1", PUPPYGRAD_CUDA_ROW_FUSION="1" if enabled else "0")
        elif args.compare == "expressions":
            env.update(PUPPYGRAD_CUDA_REDUCTIONS="1", PUPPYGRAD_CUDA_ROW_FUSION="1", PUPPYGRAD_CUDA_EXPRESSIONS="1" if enabled else "0")
        cmd = [str(runner), str(args.model_dir), str(args.source), ".cache/pup/cuda"]
        if mode:
            cmd.append(mode)
        worker = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        workers.append(worker)
        return worker

    def call(worker, **command):
        worker.stdin.write(json.dumps(command) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited: {worker.poll()}")
        return json.loads(line)

    def stop(worker):
        worker.stdin.close()
        worker.wait(timeout=30)
        assert worker.returncode == 0, worker.returncode
        workers.remove(worker)

    def cpu_ms(worker):
        # Linux process user + system CPU time. Blocked time is excluded;
        # any driver busy-waiting is counted as CPU time.
        # Tick resolution is coarse, so measure across all paired requests.
        fields = Path(f"/proc/{worker.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) * 1000 / os.sysconf("SC_CLK_TCK")

    result = {"source": str(args.source), "dtype": "F32", "kv_capacity": 512,
              "warmups": args.warmups, "repeats": args.runs, "max_new_tokens": args.tokens,
              "method": "paired shuffled sequential requests through two resident Model.infer FFI workers; compilation/loading/warmup excluded",
              "comparison": args.compare, "forwards": [], "generation": []}
    rng = random.Random(42)
    try:
        worker = start(enabled=False)
        base = call(worker, prompt="Explain why the sky is blue in one short sentence.")["tokens"]
        result["prompt_tokens"] = base
        baseline_profiles = {}
        for n in [len(base), 64, 128]:
            tokens = (base * ((n + len(base) - 1) // len(base)))[:n]
            baseline_profiles[n] = call(worker, tokens=tokens, save=str(out / f"{names[0]}-{n}.f32"))["profile"]
        stop(worker)
        worker = start(enabled=True)
        for n in [len(base), 64, 128]:
            tokens = (base * ((n + len(base) - 1) // len(base)))[:n]
            call(worker, tokens=tokens, save=str(out / f"{names[1]}-{n}.f32"))
            a, b = [array.array("f", (out / f"{name}-{n}.f32").read_bytes()) for name in names]
            assert len(a) == len(b)
            error = max(abs(x-y) for x, y in zip(a, b))
            if args.compare == "replay":
                assert a.tobytes() == b.tobytes(), (n, error)
            else:
                assert error < .003, (n, error)
            assert max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
            warm = call(worker, tokens=tokens)
            assert warm["profile"]["run_graph_builds"] == 0
            assert warm["profile"]["run_graph_launches"] == 1
            assert warm["profile"]["run_direct_kernel_launches"] == 0
            assert warm["profile"]["run_allocations"] == 0
            result["forwards"].append({"input_tokens": n, "logits_max_abs_error": error, names[0]+"_profile": baseline_profiles[n], names[1]+"_profile": warm["profile"]})
        next_token = call(worker, tokens=base)["argmax"]
        call(worker, tokens=[next_token], retain=True)
        call(worker, tokens=base)
        decode = call(worker, tokens=[next_token], retain=True)
        result["optimized_decode_profile"] = decode["profile"]
        assert decode["profile"]["run_allocations"] == 0
        assert decode["profile"]["run_graph_builds"] == 0
        assert decode["profile"]["run_graph_launches"] == 1
        assert decode["profile"]["run_input_uploaded_bytes"] == 4
        stop(worker)
        pair = {names[0]: start("ffi", False), names[1]: start("ffi", True)}
        for n in [len(base), 128]:
            tokens = (base * ((n + len(base) - 1) // len(base)))[:n]
            expected = call(pair[names[0]], tokens=tokens, count=args.tokens)["tokens"]

            def generate(name):
                sample = call(pair[name], tokens=tokens, count=args.tokens)
                assert sample["tokens"] == expected, (name, n, sample["tokens"], expected)
                return sample["token_ms"]

            for _ in range(args.warmups):
                for name in pair:
                    generate(name)
            samples = {name: [] for name in pair}
            cpu_before = {name: cpu_ms(worker) for name, worker in pair.items()}
            for _ in range(args.runs):
                order = list(pair)
                rng.shuffle(order)
                for name in order:
                    samples[name].append(generate(name))
            timings = {name: {"first_token": stats([x[0] for x in seq]),
                              "decode_token": stats([statistics.mean(x[1:]) for x in seq]),
                              "total": stats([sum(x) for x in seq])} for name, seq in samples.items()}
            row = {"input_tokens": n, "generated_tokens": len(expected), "token_ids": expected, "timings": timings,
                   "cpu_ms_per_request": {name: (cpu_ms(worker) - cpu_before[name]) / args.runs for name, worker in pair.items()},
                   "cpu_clock_tick_ms": 1000 / os.sysconf("SC_CLK_TCK"),
                   "decode_speedup": timings[names[0]]["decode_token"]["median_ms"] / timings[names[1]]["decode_token"]["median_ms"]}
            result["generation"].append(row)
            (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"input_tokens": n, "decode_speedup": row["decode_speedup"], "median_decode_ms": {name: t["decode_token"]["median_ms"] for name, t in timings.items()} }), flush=True)
        with (args.model_dir / "model.safetensors").open("rb") as checkpoint:
            result["checkpoint_sha256"] = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        result["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip()
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for worker in workers[:]:
            if worker.poll() is None:
                worker.stdin.close()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


if __name__ == "__main__":
    main()
