"""Measure retained Qwen3 speed and GPU memory as KV capacity grows."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def timings(samples):
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/qwen3-0.6b"))
    parser.add_argument("--source", type=Path, default=Path("examples/qwen3_cached.pup"))
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/qwen3-context"))
    parser.add_argument("--capacities", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--prefill-runs", type=int, default=3)
    parser.add_argument("--decode-runs", type=int, default=11)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    limit = config["max_position_embeddings"]
    if (args.chunk < args.warmups + args.decode_runs or args.chunk % 16
            or args.warmups < 3 or args.prefill_runs < 1 or args.decode_runs < 1
            or any(c <= args.chunk or c % args.chunk or c > limit for c in args.capacities)):
        parser.error("require valid capacities divisible by chunk, >=3 warmups and room for decode")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PUPPYGRAD_NVRTC", str(Path(".cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so.12").resolve()))
    deps = Path("target/release/deps")
    runner = out / "runner"
    command = ["rustc", "--edition=2021", "-O", str(Path(__file__).with_name("runner.rs")),
               "-L", f"dependency={deps}", "-o", str(runner)]
    for name in ["puppygrad", "serde_json", "tokenizers"]:
        lib = max(deps.glob(f"lib{name}-*.rlib"), key=lambda p: p.stat().st_mtime)
        command += ["--extern", f"{name}={lib}"]
    subprocess.run(command, check=True)
    runner_args = [str(runner), str(args.model_dir), str(args.source), ".cache/pup/cuda"]
    result = {"source": str(args.source), "dtype": "F32", "kv_dtype": "F32",
              "checkpoint_config_position_limit": limit, "chunk_tokens": args.chunk,
              "warmups": args.warmups, "prefill_runs": args.prefill_runs, "decode_runs": args.decode_runs,
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                               "--format=csv,noheader"], text=True).strip(),
              "method": "one fresh CUDA runtime per capacity; synthetic repeated chat tokens; warm Executable.run includes uploads and host logits; loading/compilation/state reset excluded; fixed capacity with benchmark-orchestrated chunked prefill; automatic production context sizing is not used",
              "rows": [], "single_shot_plans": []}

    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def plan(capacity, length):
        return json.loads(subprocess.check_output(runner_args + ["plan", str(capacity), str(length)], text=True))

    for capacity in sorted(set(args.capacities + [16384, 32768])):
        if capacity <= limit:
            row = {"capacity": capacity, "input_tokens": capacity - args.chunk,
                   **plan(capacity, capacity - args.chunk)}
            result["single_shot_plans"].append(row)
            print("single-shot plan", row, flush=True)
    save()

    def call(worker, **command):
        worker.stdin.write(json.dumps(command) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited with {worker.poll()}")
        return json.loads(line)

    def memory(worker):
        fields = {}
        for line in Path(f"/proc/{worker.pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                fields[key.rstrip(":")] = int(value) * 1024
        entries = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                          "--format=csv,noheader,nounits"], text=True)
        fields["gpu_process_bytes"] = None
        for line in entries.splitlines():
            pid, mib = [s.strip() for s in line.split(",")]
            if pid == str(worker.pid) and mib.isdigit():
                fields["gpu_process_bytes"] = int(mib) * 1024 * 1024
        return fields

    def decode(worker, seed):
        for _ in range(args.warmups):
            response = call(worker, tokens=[seed], retain=True)
            seed = response["argmax"]
        samples = []
        for _ in range(args.decode_runs):
            response = call(worker, tokens=[seed], retain=True)
            assert response["compile_ms"] == 0
            profile = response["profile"]
            assert profile["run_allocations"] == profile["run_graph_builds"] == 0
            assert profile["run_graph_launches"] == 1
            assert profile["run_input_uploaded_bytes"] == 4
            samples.append(response["elapsed_ms"])
            seed = response["argmax"]
        return {"timings": timings(samples), "profile": profile}

    for capacity in args.capacities:
        worker = subprocess.Popen(runner_args + [str(capacity)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        row = {"capacity": capacity, "input_tokens": capacity - args.chunk}
        try:
            base = call(worker, prompt="Explain why the sky is blue in one short sentence.")["tokens"]
            prefix = (base * ((row["input_tokens"] + len(base) - 1) // len(base)))[:row["input_tokens"]]
            for _ in range(args.warmups):
                response = call(worker, tokens=prefix[:args.chunk])
            row["early_decode"] = decode(worker, response["argmax"])
            samples, wall_samples = [], []
            expected = None
            logits_path = out / f"chunked-{capacity}.f32"
            for _ in range(args.prefill_runs):
                started, elapsed = time.perf_counter(), 0
                for offset in range(0, len(prefix), args.chunk):
                    command = {"tokens": prefix[offset:offset + args.chunk], "retain": offset > 0}
                    if offset + args.chunk == len(prefix):
                        command["save"] = str(logits_path)
                    response = call(worker, **command)
                    assert response["compile_ms"] == 0
                    assert response["profile"]["run_allocations"] == 0
                    elapsed += response["elapsed_ms"]
                wall_samples.append((time.perf_counter() - started) * 1000)
                samples.append(elapsed)
                logits = logits_path.read_bytes()
                if expected is not None:
                    assert logits == expected, "reset/repeated prefill must produce identical logits"
                expected = logits
            row["prefill"] = {"timings": timings(samples), "wall_timings": timings(wall_samples),
                              "profile": response["profile"], "memory": memory(worker)}
            row["decode"] = decode(worker, response["argmax"])
            row["decode"]["memory"] = memory(worker)
            if capacity == max(args.capacities):
                alternate = args.chunk * 2
                # Compile/warm the alternative shape before checking chunk invariance.
                call(worker, tokens=prefix[:alternate])
                alternate_path = out / f"alternate-{capacity}.f32"
                for offset in range(0, len(prefix), alternate):
                    command = {"tokens": prefix[offset:offset + alternate], "retain": offset > 0}
                    if offset + alternate >= len(prefix):
                        command["save"] = str(alternate_path)
                    response = call(worker, **command)
                a, b = [array.array("f", p.read_bytes()) for p in [logits_path, alternate_path]]
                error = max(abs(x - y) for x, y in zip(a, b))
                assert error < .003 and max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
                row["chunk_invariance"] = {"alternate_chunk_tokens": alternate, "logits_max_abs_error": error}
            print("measured", capacity, "prefix", len(prefix), "prefill_ms", row["prefill"]["timings"]["median_ms"],
                  "decode_ms", row["decode"]["timings"]["median_ms"], "state_bytes", row["decode"]["profile"]["state_bytes"], flush=True)
        except Exception as error:
            row["error"] = str(error)
            print("failed", capacity, error, flush=True)
        finally:
            worker.stdin.close()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            row["worker_exit_code"] = worker.returncode
            result["rows"].append(row)
            save()

    # Separate runtime so the much larger single-shot arena does not inflate
    # the chunked benchmark's memory counters.
    capacity = max((r["capacity"] for r in result["single_shot_plans"] if "error" not in r["result"]), default=None)
    if capacity in args.capacities:
        worker = subprocess.Popen(runner_args + [str(capacity)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        try:
            length = capacity - args.chunk
            base = call(worker, prompt="Explain why the sky is blue in one short sentence.")["tokens"]
            prefix = (base * ((length + len(base) - 1) // len(base)))[:length]
            for _ in range(args.warmups):
                call(worker, tokens=prefix)
            samples = []
            for _ in range(args.prefill_runs):
                response = call(worker, tokens=prefix, save=str(out / f"single-shot-{capacity}.f32"))
                assert response["compile_ms"] == response["profile"]["run_allocations"] == 0
                samples.append(response["elapsed_ms"])
            a, b = [array.array("f", (out / f"{mode}-{capacity}.f32").read_bytes()) for mode in ["chunked", "single-shot"]]
            error = max(abs(x - y) for x, y in zip(a, b))
            assert error < .003 and max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
            result["single_shot"] = {"capacity": capacity, "input_tokens": length, "timings": timings(samples),
                                     "profile": response["profile"], "memory": memory(worker),
                                     "logits_max_abs_error_against_chunked": error}
            print("single-shot measured", result["single_shot"], flush=True)
        finally:
            worker.stdin.close()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            save()
    with (args.model_dir / "model.safetensors").open("rb") as checkpoint:
        result["checkpoint_sha256"] = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    save()
    if any("error" in r for r in result["rows"]):
        raise RuntimeError("one or more capacities failed; see results.json")


if __name__ == "__main__":
    main()
