# Qwen3-0.6B CUDA comparison

Run from the repository root, after building with `cargo build --release` and
placing the official Qwen3-0.6B checkpoint in `models/qwen3-0.6b` (see
[model instructions](../../docs/qwen3.md)). This benchmark requires Python with
NumPy, a tinygrad checkout, the NVIDIA driver, and NVRTC. Python only drives the
comparison and tinygrad; Puppygrad executes its generated CUDA kernels in a Rust
worker linked against the production release library.

```bash
PUPPYGRAD_NVRTC=/path/to/libnvrtc.so.12 \
NVRTC_PATH=/path/to/libnvrtc.so.12 \
python benchmarks/qwen3-cuda/benchmark.py --tinygrad-root /path/to/tinygrad
```

The measured tinygrad revision was
`1a58c3ae9d5ff5605d81085cf1a364c113e95cf6`. It supports this architecture through
`tinygrad.llm.model.Transformer`; `examples/llm.py` is not its current entrypoint.
The CLI lists `qwen3:0.6b`, normally backed by a Q8_0 GGUF. This script instead
maps the same official safetensors to the native model, expanding BF16 weights to
F32 in both implementations. It also sets tinygrad's KV cache to F32 (its default
is F16). No model operations are rewritten for the comparison.

Full-prefix trials overwrite tinygrad's cache from position zero, and include
input upload and final logits download on both sides. The Rust timer surrounds
`Executable.run`; tinygrad is measured through `TinyJit` replay, using `BEAM=0`.
Tokenization, weight loading, graph compilation and warmup are excluded. Trials
alternate in shuffled order. Results also compare native cached generation with
Puppygrad generation. The default source uses full prefixes; select the retained
source below for a cache-to-cache comparison.
Generation runs through the actual Puppygrad `Model.infer` FFI, including token
binding, host sampling and streaming callbacks, versus tinygrad's native
`Transformer.generate`, including GPU sampling and token fetch. Each new tinygrad
generation resets prefix reuse; its context capacity is 512 and chunk size is 32.

The harness checks full logits and eight greedy token IDs before reporting
measurements. JSON includes samples, logit errors, compilation times, GPU kernel
counts and retained-memory counters. Output defaults to the ignored
`.cache/qwen3-compare/` directory; choose another with `--output-dir`.

For retained Qwen generation, with equal F32 weights and 512-row F32 KV capacity:

```bash
python benchmarks/qwen3-cuda/benchmark.py \
  --source examples/qwen3_cached.pup --output-dir .cache/qwen3-cached
```

Each direct forward resets state and starts at position zero. Generation parity
also checks incremental executions (prompt, then one token per call) before
measuring the real FFI. Retained-state bytes, allocations and token-only upload
counters are recorded in `puppygrad_generation_profile`. The FFI first-token timer
includes a reset; direct kernel execution timers exclude it. Tinygrad's native
32-token chunk size and symbolically sized prefill can affect its first-token
latency separately from steady decode. Cold compilation and model loading are
excluded from warm performance comparisons and must be assessed separately.
