# Puppygrad

Rust-first deep learning sandbox with:
- an in-house tinygrad-inspired computation engine, and
- a `qwen` CLI path for running Qwen2.5-0.5B on CPU.

## Build

```bash
cargo build --release
```

## Run Qwen 0.5B (CPU)

```bash
./target/release/puppygrad qwen \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain RoPE in simple words." \
  --max-new-tokens 120
```

or from local files:

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
- No transformer runtime yet; this phase focuses on building our own core engine first.

