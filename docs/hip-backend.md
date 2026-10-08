# HIP backend

Puppygrad emits HIP C++, compiles it to AMD code objects with HIPRTC, and executes
it through the dynamically loaded ROCm HIP runtime. CPU, CUDA and HIP use the
same Pop graphs, view indexing, contraction/reduction schedules and buffer
ownership. This follows AMD's [HIPRTC module execution path](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html).
Model math stays in `.pup`; no Python, rocBLAS or external tensor framework is
needed for this backend.

## Requirements

Execution requires 64-bit Linux, a ROCm-supported AMD GPU with accessible
`/dev/kfd` and render devices, ROCm 6 or newer, `libamdhip64` and `libhiprtc`.
CPU builds and HIP source emission do not load ROCm libraries. `hipcc` and toolkit
headers are not needed for runtime compilation: HIPRTC supplies its device
headers. The runtime queries the selected GPU's versioned device properties for
its `gfx...` architecture and asks HIPRTC to compile for that target. An
unsupported architecture or missing device fails explicitly without CPU fallback.

Standard library names and `/opt/rocm/lib` / `/opt/rocm/lib64` are searched.
Custom installations can set:

```sh
export PUPPYGRAD_HIP_RUNTIME=/absolute/path/to/libamdhip64.so
export PUPPYGRAD_HIPRTC=/absolute/path/to/libhiprtc.so
```

Keep the matching ROCm dependencies available to the dynamic loader.

## Examples

`hip` selects device zero; `hip:N` selects HIP ordinal N. The Qwen checkpoint
and `.pup` source are the same ones used by CPU and CUDA:

```sh
cargo run --release -- llm examples/qwen3_cached.pup \
  --model-dir models/qwen3-0.6b --device hip:0 \
  --prompt 'What is a compiler?' --temperature 0 --stream

cargo run --release -- llm examples/qwen3.pup \
  --model-dir models/qwen3-0.6b --device hip:0 \
  --prompt 'What is a compiler?' --temperature 0 --stream

cargo run --release -- llm examples/llm.pup \
  --model-dir models/gpt2 --device hip:0 \
  --prompt 'The meaning of life is' --max-new-tokens 8 --verify-reference

cargo run --release -- llm capacity examples/qwen3_cached.pup \
  --model-dir models/qwen3-0.6b --device hip:0 --json
```

The cached Qwen program retains KV state on the GPU, resets it between requests,
and shares weights/state between prefill and decode shapes. Automatic context
planning, memory preflight checks and chunked prefill use HIP's live free memory
and its actual selected allocation schedules. Capacity planning with
`--memory-budget-mib` works without a GPU or ROCm installation.

Training and image stage providers accept both HIP and CUDA devices:

```sh
cargo run --release -- train examples/mnist.pup --device hip:0 --epochs 5
cargo run --release -- train examples/mnist_autodiff.pup --device hip:0 --epochs 5
cargo run --release -- image examples/image.pup --device hip:0 --download \
  --prompt 'a puppy in a sunny garden' --width 256 --height 256 --output puppy.png
```

The training harness retains its existing host parameter/checkpoint contract:
updated parameter outputs return to the host and upload on the next batch.
GPU training reports synchronized host-call timings, including transfers; it
omits device kernel/packing counters because GPU event profiling is unavailable.
Generated source is saved as `program.hip` (or `program.cu` for CUDA) alongside
metrics and checkpoints. Image stages use independent GPU runtimes because
their input/weight slot numbers overlap; weights remain resident within each
stage. Intermediate image-stage outputs currently return through host tensors.
`--cpu-target` applies only to CPU execution.

Matmul and linear programs can be emitted without hardware:

```sh
cargo run -- emit examples/matmul.pup --backend hip --raw -o matmul.hip
cargo run -- emit examples/linear.pup --backend hip --raw -o linear.hip
cargo run -- emit examples/qwen3_cached.pup --backend hip \
  --model-dir models/qwen3-0.6b --sequence-length 5 --raw -o qwen3.hip
```

Their device kernels use the same buffer/launch contract as CUDA. The files do
not include a host executable. Use `compiler::hip::compile` and
`Executable::run` to run arbitrary tensor graphs with caller-supplied inputs.
`hip::Runtime` and `hip::compile_with_runtime` share residency between shapes.
`hip::compile_source(source, "gfx1100")` compiles emitted source to an ELF code
object with HIPRTC without an attached GPU.

## Scheduling and ownership

HIP reuses shared-memory matmul tiling, pointwise/epilogue fusion, row-local RMS
normalization and stable softmax, checked gather/store bounds, lifetime-based
arena planning and retained state. Larger matrices use padded 32x64 tiles with
eight output accumulators per thread. Softmax rows above width 4096 stream their
values instead of keeping large per-lane arrays or separate exponential tensors;
row fusion supports widths through 65536. F32 reductions and small contractions
operate in explicit 32-lane groups. HIP shuffle calls specify width 32, so adjacent groups
stay independent on both wave32 and wave64 devices. F32, I32, U8 and Bool buffers
are supported. Floating-point contraction and fast math are disabled; numerical
comparisons still use tolerances for GPU transcendental and reduction differences.

HIP uses its own pointer, error-string and graph parameter ABIs. Primary contexts
are retained and pushed/popped around execution and cleanup; executables cannot
implicitly cross Rust threads. Buffers and modules are freed when their final
owners drop. Immutable Arc-backed inputs skip repeated uploads until replaced
or modified through copy-on-write. Warm executions reuse allocation capacity.

Ordered HIP graphs replay by default. Allocation or state-layout changes
invalidate cached graph addresses; resets and updated input contents reuse them.
Set `PUPPYGRAD_HIP_GRAPH=0` or `Runtime::set_graph_replay(false)` for direct
launches. The shared runtime exposes allocation/upload and submission counters.
HIP graph nodes use AMD's [HIP graph API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html).

Set `PUPPYGRAD_HIP_REGISTER_TILES=0` to use the previous 16x16 matrix tiles.
Set `PUPPYGRAD_HIP_REDUCTIONS=0`, `PUPPYGRAD_HIP_ROW_FUSION=0`, or
`PUPPYGRAD_HIP_EXPRESSIONS=0` before compilation to disable the corresponding
schedules. CUDA's switches affect CUDA only. Generated `.hip` and `.hsaco` files
are cached under `.cache/pup/hip`, keyed by source, backend, architecture and RTC
major/minor version. The shared `./puppygrad.db` (overridden with `PUPPYGRAD_DB`) records
paths, SHA-256 hashes, sizes, compatibility and module-load statistics; source and
binary files stay on disk. See [kernel cache bookkeeping](llm-runtime.md) for
fields and an inspection query. Tensor cores, GPU event profiling and provider-library export
are not implemented; `emit --backend hip --profile` reports an explicit error.
The thread-scaling LLM benchmark remains CPU-only.

## Verification

Ordinary tests cover source emission, selectors, CLI validation, memory plans,
and the HIP FFI layouts. HIPRTC compilation tests require libraries but no GPU:

```sh
cargo test --test compiler_hip --test qwen3 hiprtc_ -- --ignored
cargo test --lib hip_runtime_symbols_load_without_a_gpu -- --ignored
```

AMD hardware checks cover CPU/HIP parity for operations, tiled/warp contractions,
convolution, gradients, bounds-error recovery, copy-on-write uploads, resident
state, graph/direct launches and Qwen logits/greedy generation:

```sh
cargo test --test compiler_hip hip_matches_cpu -- --ignored --test-threads=1
cargo test --test compiler_hip hip_bounds -- --ignored --test-threads=1
cargo test --test compiler_hip hip_mnist -- --ignored --test-threads=1
cargo test --test compiler_hip hip_register_tiles -- --ignored --test-threads=1
cargo test --test compiler_hip hip_wide_softmax -- --ignored --test-threads=1
cargo test --test compiler_state hip_state -- --ignored --test-threads=1
cargo test --test qwen3 hip_ -- --ignored --test-threads=1
cargo test --test training_runtime hip_harness -- --ignored --test-threads=1
cargo test --lib hip_dropping -- --ignored --test-threads=1
# These also require the stable-diffusion-tiny checkpoint:
cargo test --test image_stages hip_ -- --ignored --test-threads=1
```

During implementation, ROCm 7.2.4 HIPRTC successfully compiled tensor operation,
matmul/linear, manual/autodiff MNIST, DDIM image-stage and Qwen full/cached
prefill/decode kernels for `gfx1100` and `gfx90a`. All runtime symbols resolved,
and the versioned property/graph layouts matched the installed AMD headers.
Hardware checks passed on an AMD Radeon RX 9070 XT using ROCm 7.2.4: matmul
and linear examples, operations/convolution/autodiff, both MNIST training graphs,
scalar training/checkpoint output, bounds recovery, input residency, graph/direct
execution, retained-state reset and final-owner cleanup. Small deterministic Qwen
checkpoints matched independent reference logits and greedy generation across
prefixes, chunked prefill and repeated requests.

The pinned full Qwen3-0.6B checkpoint generated 32 tokens through the optimized
HIP CLI with `examples/qwen3_cached.pup`. Its transcript is in
`.cache/hip-validation/qwen3-cached.stdout` / `.stderr`. This confirms the real
model path; it is not a controlled performance benchmark.

To measure warm production Qwen generation across context sizes, through the
live capacity estimate:

```sh
cargo build --release
python3 benchmarks/qwen3-hip/generation.py --output-dir .cache/hip-context
```

This uses a fresh production LLM FFI worker for each size, F32 weights/KV,
128-token prefill chunks and repeated chat token IDs. Each row leaves room for
32 generated tokens. A separate config disables the EOS stop condition so every
request generates the same number of tokens; original checkpoint files and
weight values stay unchanged. One full request warms each worker before three
measured requests. Timers include reset, prefill, host sampling, transfers and
streaming callbacks, and exclude loading and compilation. Decode throughput
excludes the first token. JSON stores every token latency and checks deterministic
greedy output; CSV stores the table. Recorded VRAM usage includes the desktop and
other processes. This measures execution at long contexts, not retrieval quality.

On 2026-10-06, before the register-tile and wide-softmax optimizations,
Qwen3-0.6B passed the complete sweep on an RX 9070 XT with ROCm 7.2.4.
All 36 requests completed and repeated requests produced identical
greedy tokens. Context below is input plus 32 output tokens; first-token and
whole-request times are medians of three warm requests. Decode speed averages
the 31 subsequent token intervals across all three requests.

| Context tokens | First token (s) | Decode (tokens/s) | 32-token request (s) |
|---:|---:|---:|---:|
| 56 | 0.021 | 123.9 | 0.272 |
| 512 | 0.231 | 124.0 | 0.481 |
| 1,024 | 0.510 | 115.7 | 0.778 |
| 2,048 | 1.235 | 100.5 | 1.543 |
| 4,096 | 3.159 | 74.0 | 3.578 |
| 8,192 | 11.027 | 39.0 | 11.822 |
| 16,384 | 37.869 | 23.0 | 39.220 |
| 32,768 | 150.214 | 10.5 | 153.155 |
| 40,960 | 204.356 | 8.8 | 207.885 |

The maximum row processed 40,928 input tokens and generated 32 more, with
15.22 GiB total device VRAM use out of 15.92 GiB reported by sysfs. Raw samples,
checkpoint/source hashes and the CSV are in `.cache/hip-context/results.json`
and `.cache/hip-context/results.csv`. This verifies execution through the
checkpoint's declared position limit, including beyond its advertised 32K window;
it does not establish model quality at those lengths.

The subsequent register-tile and streaming-softmax changes were measured with
the same release build configuration, F32 checkpoint, prompt tokens, 128-token
chunks and 32-token output length. Three warm production requests per row gave:

| Context tokens | Previous first token (s) | Optimized first token (s) | Speedup | Optimized decode (tokens/s) | Optimized 32-token request (s) |
|---:|---:|---:|---:|---:|---:|
| 56 | 0.0213 | 0.0212 | 1.01x | 124.2 | 0.271 |
| 8,192 | 11.027 | 6.661 | 1.66x | 38.5 | 7.466 |
| 40,960 | 204.356 | 126.790 | 1.61x | 8.3 | 130.526 |

Every generated token matched the baseline in all warmup and measured requests.
Separate 128-token chunk measurements at capacities 512, 8192 and 40960 produced
bitwise-identical full F32 logits; median chunk times improved from 61.72 to
37.16 ms, 170.78 to 104.53 ms and 627.16 to 388.66 ms, respectively. GPU tests
also checked matrix layouts, ragged tiles, fused epilogues and wide softmax with
NaNs, infinities and signed zero against CPU output, with graph and direct matrix
launches. HIPRTC compiled the matrix fixtures for both `gfx1100` and `gfx90a`.

Production samples and hashes are in `.cache/hip-prefill-optimized/results.json`;
the before/after table is in `.cache/hip-prefill-optimized/comparison.csv`.
Chunk samples and logits are in `.cache/hip-prefill-optimization`.
Baseline and optimized production measurements were taken at different times
on the same GPU. Decode throughput was slightly lower in the later run
(39.0 to 38.5 tokens/s at 8K, 8.8 to 8.3 at maximum context); these changes
primarily improve prefill. The attention kernels still scan the full reserved
KV capacity, including masked future positions. Avoiding those reads and
combining attention operations remain opportunities for further improvement.

The public tiny Stable Diffusion fixture passed CLIP, UNet and VAE comparisons
against the native reference (maximum absolute error below 0.0000016), repeated
image requests, invalid-input recovery and resolution changes. The image CLI
saved `.cache/hip-validation/tiny-sd-hip.png`. This fixture has random weights and
checks execution rather than image quality. Full SD1.5 checkpoint inference and
controlled performance comparisons remain unmeasured.

The sandbox hides GPU devices in this environment. Hardware validation was run
with GPU access outside the sandbox; ordinary source tests and offline HIPRTC
compilation do not need that access.
