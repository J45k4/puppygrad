# Puppygrad

Rust-first deep learning sandbox with:
- an in-house tinygrad-inspired computation engine, and
- native model runtime experiments built directly in Rust.

## Build

```bash
cargo build --release
```

## Models

Implemented:

| Model | Status | Runtime | Notes |
| --- | --- | --- | --- |
| GPT-2 | Working | Rust reference | Loads Hugging Face `config.json`, `tokenizer.json`, and `model.safetensors`; uses greedy/sampled decoding, token streaming, and a KV cache. |
| Qwen | Stub | None yet | CLI placeholder for future native loading/runtime work. |

Model assets are stored under the project-root `models/` directory, which is ignored by git. Rust source lives under `src/models/` and is tracked. GPT-2-specific code is organized under `src/models/gpt2/`, with the current Rust reference implementation in `src/models/gpt2/rust.rs`.

### Run GPT-2 small

First run downloads GPT-2 small assets into `models/gpt2`:

```bash
./target/release/puppygrad gpt2 \
  --download \
  --backend rust \
  --threads 4 \
  --stats \
  --prompt "Hello, my name is" \
  --max-new-tokens 20
```

After assets are downloaded, `--download` is optional:

```bash
./target/release/puppygrad gpt2 \
  --backend rust \
  --threads 4 \
  --stats \
  --prompt "The future of GPU compilers is" \
  --max-new-tokens 20
```

Use a different GPT-2-family checkpoint by giving both a model id and local directory:

```bash
./target/release/puppygrad gpt2 \
  --download \
  --model-id gpt2-medium \
  --model-dir models/gpt2-medium \
  --backend rust \
  --threads 4 \
  --prompt "Rust makes systems programming" \
  --max-new-tokens 20
```

The GPT-2 runtime is intentionally simple: CPU `f32` and no GPU kernels yet. The only backend today is `rust`; `--threads` controls puppygrad's own thread pool. GPT-2 runs print generated-token throughput to stderr after generation.

Greedy decoding is the default. For less repetitive text, enable sampling and repeat penalty:

```bash
./target/release/puppygrad gpt2 \
  --prompt "Hello, my name is" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  --repeat-last-n 128 \
  --seed 42
```

When `models/gpt2/puppygrad-tune.json` exists, the `gpt2` command loads it automatically. Explicit CLI flags override the saved config. Use `--no-tuning` to ignore the saved file, or `--tuning-file path/to/tune.json` to load a different file.

Pass `--stats` to print the full performance breakdown to stderr while streamed text stays on stdout. The current GPT-2 stats include model load time, tokenization time, prefill time, time to first token, decode time, average decode-token latency, and token/sec rates for prefill, decode, and total model tokens.

### Run GPT-2 experiments

Sweep backend settings and print averaged performance rows:

```bash
./target/release/puppygrad experiment gpt2 \
  --threads 1,2,4,8 \
  --max-new-tokens 16,32,64 \
  --runs 5 \
  --warmup-runs 1 \
  --prompt "The future of GPU compilers is"
```

Use `--format csv` or `--format json` when you want to plot results or compare runs outside the terminal.

### Autotune GPT-2 settings

Search candidate backend settings and print the fastest measured config:

```bash
./target/release/puppygrad autotune gpt2 \
  --threads 1,2,4,8,12,16,24,32 \
  --max-new-tokens 16 \
  --runs 2 \
  --warmup-runs 1 \
  --max-trials 48 \
  --prompt "The future of GPU compilers is"
```

The autotuner is generic internally: a target provides candidate configs, a trial runner, and a score. GPT-2 currently scores candidates by generated-token decode throughput.

By default, GPT-2 autotune saves the best config to `models/gpt2/puppygrad-tune.json`. Pass `--save-tuning path/to/file.json` to choose another location.

## Qwen Runtime Placeholder

```bash
./target/release/puppygrad qwen \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain RoPE in simple words." \
  --max-new-tokens 120
```

The `qwen` command is currently a stub. The previous external runtime was removed so model work can focus on low-level implementation inside this project.

The CLI still accepts local model paths for future native loading work:

```bash
./target/release/puppygrad qwen \
  --model-dir ./models/qwen2.5-0.5b-instruct \
  --prompt "Write a short Rust tip." \
  --max-new-tokens 120
```

## Run linear regression demo

```bash
./target/release/puppygrad demo-linear --steps 300 --lr 0.1
```

## Run matmul backward check

```bash
./target/release/puppygrad matmul-check
```

## Engine status

Implemented:

- Dynamic computation graph with reverse-mode autodiff.
- Scalar + vector operations: `add`, `sub`, `mul`, `relu`, `tanh`, `sum`, `mean`.
- 2D `matmul` forward and backward.
- Scalar broadcast support in binary ops.
- SGD parameter update helpers (`zero_grad`, `step`).

Current limits:

- CPU only.
- No advanced broadcasting rules beyond scalar broadcast.
- No kernel fusion or SIMD tuning yet.
- The GPT-2 runtime is a reference implementation, not optimized tensor infrastructure.
