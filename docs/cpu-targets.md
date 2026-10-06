# CPU targets

The C backend emits GCC/Clang C vector microkernels and uses automatic
vectorisation for other loops. `.pup` programs do not change. Generic targets
use 128-bit vectors; AVX2-enabled targets use 256-bit vectors.

```sh
target/release/puppygrad train examples/mnist_autodiff.pup --cpu-target native
target/release/puppygrad llm examples/llm.pup --cpu-target avx2 --threads 1
target/release/puppygrad llm benchmark examples/llm.pup --cpu-target native --max-threads 1
```

- `generic` (default): use the C compiler's default instruction target.
- `native`: add `-march=native`, using this host's CPU features and tuning.
- `avx2`: add `-mavx2`, requiring an AVX2-capable x86 host for execution.

All targets use `-O2 -fno-math-errno -fwrapv -ffp-contract=off`; there is no
fast-math or fused multiply-add enabled by the target switch. These options
select compiler instructions; threading is controlled separately. Already built
shared-library providers cannot be retargeted through this switch.

`emit` produces the same C source for every target, using GCC/Clang vector
extensions and conditional vector widths. To compile it yourself, add the
desired target flag to the [documented C compilation command](compiler-ir.md).
The Rust APIs `compile_with_options` / `compile_profiled_with_options` accept
`cpu::BuildOptions { cpu_target: cpu::CpuTarget::Native }`; existing APIs default
to `Generic`.

## Cache and telemetry

Cache keys cover generated source, target, compiler driver path/identity, target
triple and flags. For `native`, effective compiler target macros and host CPU
features form an additional fingerprint. A JSON build manifest must match before
loading a cached library. Existing entries without a manifest rebuild once.

`Executable::build_info`, training metrics and LLM benchmark worker results
record these build settings. Targeted artifacts are host artifacts; copying a
native `.so` elsewhere does not make it portable.

## Local measurements before packed GEMM lowering

Measured on an i7-8700K with GCC 16.2.1, one thread pinned to CPU 0. After five
warmups per target, 21 rounds interleaved targets in a fixed-seed random order.
MNIST trials averaged ten calls; GPT-2 trials used one call. Reported times are
medians across trials. Loading, compilation and profile aggregation are excluded;
caller output allocation, FFI and C profiling are included. The release build
finished before measurements started.

| Target | MNIST training step, batch 128 | GPT-2 forward, 5-token prefix |
| --- | ---: | ---: |
| generic | 2.120 ms | 248.430 ms |
| avx2 | 0.966 ms (2.20x) | 201.170 ms (1.24x) |
| native | 0.976 ms (2.17x) | 188.108 ms (1.32x) |

MNIST uses `examples/mnist_autodiff.pup`, the first 128 official training rows
and a fixed trained checkpoint. GPT-2 uses the real local checkpoint and prompt
IDs `[464, 3616, 286, 1204, 318]`. These are fixed training-step / forward-pass
comparisons, not full epoch or autoregressive generation timings. All returned
floats matched the generic target bit for bit. Tensor arena sizes were unchanged:
72,448 bytes for MNIST and 107,548 bytes for GPT-2. Matmul workers separately
declare 1,792 bytes of tile scratch.

Disassembly showed 256-bit packed multiply instructions for `avx2` and `native`
in both models, with no fused multiply-add instructions. The two largest MNIST
contractions remain its main kernel costs; the GPT-2 vocabulary projection is its
largest individual kernel. This supports investigating contraction layouts and
tiling before introducing explicit SIMD language operations.

A full five-epoch native MNIST run reached 95.45% test accuracy; its final
checkpoint was byte-identical to the generic run. Native GPT-2 CLI generation
also passed the Rust reference check for two generated tokens (maximum absolute
logit errors 0.000275 and 0.000290, identical greedy choices).

The reproducible local measurement driver and raw samples/profiles are under
ignored `.cache/cpu-targets/{benchmark.rs,results.json}`. The native training run
is in `.cache/train/mnist-autodiff-native-5epochs/`.

## Generic contraction optimizations

Matrix contractions use 4-by-8 register tiles on generic targets and 4-by-16
register tiles on AVX2 targets. Eligible strided contractions with a reduction
dimension of at least 1024 use 128-by-16 output macro tiles and 512-element
reduction chunks. Each active worker packs and consumes its own 32 KiB panel
without barriers between chunks. Up to 16 workers share one bounded allocation,
at most 512 KiB in total. The worker count also respects the old panel capacity
for smaller reductions. Partial sums use the existing destination buffer; fused
epilogues run only after the final chunk. Sequential addition order and
`-ffp-contract=off` are retained.

Smaller eligible contractions use a shared 64-column panel packed serially.
Small contractions, and contractions with known contiguous or broadcast weight
columns, read operands directly. Packing is invocation-local, including training
updates; no duplicate model weights are retained and no full convolution patch
matrix is allocated.

Coordinates propagate through contiguous reshape groups and other views, keeping
spatial indices independent of reduction indices. Large independent elementwise,
padding, and reduction output loops share the contraction worker pool; small
loops stay serial. These changes apply to arbitrary matching Pop graphs.

Profiling separates arena storage (`workspace_bytes`) from the maximum live
packed allocation across workers (`packed_workspace_bytes`). Both, along with
caller outputs, count toward the 2 GiB workspace limit. Declared GEMM array scratch is at most 320 bytes
per worker; compiler spill slots and worker stacks are separate.

Kernel `packed_bytes` records the packed allocation capacity per invocation;
`packed_bytes_total` sums that capacity over calls, rather than reporting memory
traffic across panel reuse or repeated output macro tiles. Kernel elapsed time
always measures wall time. For cache-blocked kernels, packing and compute overlap
across workers: `phase_timing_kind: "worker_average"` marks phase counters as
average per-worker elapsed time. Other kernels use `"wall"` phase counters.

### SD1.5 paired measurement

On the same i7-8700K, the old and new C libraries ran a complete 256-by-256,
20-step float32 SD1.5 pipeline with eight threads and the native target. Both
shared the same checkpoint arrays, prompt tokens, initial noise and scheduler;
execution order alternated at each timestep. Compilation, loading and stage
warmups were excluded; profiling was enabled for both.

| Work | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| CLIP, both prompts | 0.820 s | 0.872 s | 0.94x |
| U-Net and DDIM, all 20 steps | 210.168 s | 176.771 s | 1.19x |
| VAE decoding | 23.932 s | 10.517 s | 2.28x |
| Complete warm pipeline | 234.920 s | 188.160 s | 1.25x |
| Median guided denoising step | 10.239 s | 8.151 s | 1.26x |

Both text contexts, every intermediate denoising latent and the final RGB8 image
matched the old C implementation bit for bit. CLIP was slightly slower in this
run; these changes do not guarantee a gain for every contraction shape. Other
builds and browser tests were active, so these are local paired observations
rather than isolated-machine guarantees or a new comparison with tinygrad.

U-Net and VAE tensor arenas remain 71.544 MiB and 225.004 MiB. Maximum live packed
panels add 5.625 MiB and 1.125 MiB respectively; they do not duplicate the full
checkpoint. MNIST's existing arena regression remains at most 70 KiB, with no
packing allocations. All 72 focused compiler/runtime checks passed, including
local SD checkpoint parity and repeated-request lifecycle tests.

The ignored `.cache/sd1/optimize/` directory contains `comparison.md`, raw
`model-results.json` with category totals and top kernel profiles, old/new images,
and the `emit.rs`, `model.py`, and `build-and-benchmark.sh` reproduction drivers.

### Parallel packing measurement

A later paired run used the preceding optimized, serial-packing libraries as
the baseline. Dimensions, weights, inputs, float32 arithmetic, eight threads,
native target and alternating execution order were unchanged.

| Work | Serial packing | Parallel packing | Speedup |
| --- | ---: | ---: | ---: |
| CLIP, both prompts | 0.610 s | 0.577 s | 1.06x |
| U-Net and DDIM, all 20 steps | 143.655 s | 105.079 s | 1.37x |
| VAE decoding | 9.560 s | 9.735 s | 0.98x |
| Complete warm pipeline | 153.825 s | 115.392 s | 1.33x |
| Median guided denoising step | 7.163 s | 5.217 s | 1.37x |

U-Net packing wall time fell from 52.637 to 16.470 seconds; compute wall time
was nearly unchanged (81.561 to 79.232 seconds). Arenas and maximum live panel
capacities were identical. Every intermediate latent and the RGB8 image again
matched bit for bit. All 72 focused checks and the release image CLI passed.

VAE was approximately unchanged: its larger compute workload already amortized
serial packing. The isolated packing-heavy U-Net kernel 479 improved 2.00x with
eight threads; kernel 572 improved 1.27x. Their single-thread performance stayed
approximately unchanged.

Host load differed between the two benchmark sessions; compare each table's
paired measurements rather than combining their absolute times or speedups.
Ignored `.cache/sd1/parallel-pack/` contains `comparison.md`, `model-results.json`,
complete `baseline-*-profile.json` and `optimized-*-profile.json` kernel counters,
`kernel-results.json`, the images, and reproduction drivers.

### Worker-local cache blocking measurement

The next paired run compared the preceding shared parallel-packing compiler with
worker-local cache blocking. SD1.5, 256-by-256 pixels, 20 float32 DDIM steps,
eight threads, native C target, identical inputs and alternating execution order
were retained. A separate `doorcontrol-rs` test process consumed approximately
ten logical cores, so these results describe a heavily contended workstation.

| Work | Shared panels | Private cache panels | Speedup |
| --- | ---: | ---: | ---: |
| CLIP, both prompts | 3.218 s | 2.930 s | 1.10x |
| U-Net and DDIM, all 20 steps | 434.388 s | 336.381 s | 1.29x |
| VAE decoding | 24.392 s | 19.831 s | 1.23x |
| Complete warm pipeline | 461.999 s | 359.142 s | 1.29x |
| Median guided denoising step | 22.776 s | 16.803 s | 1.36x |

Text contexts, every denoising latent and the RGB8 image again matched bit for
bit. Arenas were unchanged. Maximum live packing storage fell from 5.625 to
0.500 MiB for U-Net and from 1.125 to 0.500 MiB for VAE; CLIP fell from 0.750
to 0.500 MiB. This is a bound on packed storage, not a separate process-RSS
comparison. Phase counters now distinguish `worker_average` from `wall`, because
packing and compute overlap across workers. Whole-kernel wall counters remain
comparable.

All 72 focused checks passed, including profiled and plain C, generic/native
targets, mutable parameters, odd macro/reduction tails, MNIST memory, and local
SD checkpoint parity. The rebuilt release image CLI smoke test also passed.
Ignored `.cache/sd1/cache-block/` holds `comparison.md`, `model-results.json`,
complete old/new profiles, images, hot-kernel samples and reproduction drivers.
Compare each session's paired values; do not combine absolute times or speedups
across these sessions with different host load.
