# CUDA backend

Puppygrad can emit CUDA C++ kernels, compile them to PTX using NVIDIA NVRTC,
and load/launch that PTX using the NVIDIA driver API. Model arithmetic is
compiler-generated GPU code. Rust manages contexts, launches, and buffer copies;
there is no Python, cuBLAS, or external tensor framework in this execution path.
This follows NVIDIA's documented [NVRTC-to-driver execution path](https://docs.nvidia.com/cuda/nvrtc/index.html).

## Commands

Emission requires no GPU or CUDA installation:

```sh
puppygrad emit examples/matmul.pup --backend cuda -o matmul.cu
puppygrad emit examples/llm.pup --backend cuda --sequence-length 5 -o llm.cu
```

Pretty printing uses the existing `clang-format` path. `--raw` skips formatting.
The emitted file contains device kernels; buffer bindings and launch order are
owned by the Rust compiler runtime. It is not a standalone host executable or
an exported model shared library.

Execution requires Linux, an NVIDIA GPU/driver, and NVRTC:

```sh
puppygrad run examples/llm.pup --device cuda:0 --model-dir models/gpt2 \
  --prompt 'The meaning of life is' --max-new-tokens 8 --verify-reference
```

`cuda` selects device 0. `cuda:N` selects driver ordinal N. Driver and compiler
libraries are loaded only for CUDA execution; CPU builds remain independent of
the CUDA toolkit. No `nvcc` command or toolkit headers are required. The runtime
queries the GPU's compute capability and asks NVRTC to compile for it. NVRTC
must support that architecture; unsupported devices/compiler versions fail explicitly.

Standard NVRTC library locations are searched. For a custom installation:

```sh
export PUPPYGRAD_NVRTC=/absolute/path/to/libnvrtc.so.12
```

Keep NVIDIA's matching `libnvrtc-builtins` beside it. This workstation's test
installation was downloaded as NVIDIA's `nvidia-cuda-nvrtc-cu12==12.6.85` wheel
and extracted into the ignored `.cache/cuda-toolchain/` directory. To use it here:

```sh
export PUPPYGRAD_NVRTC="$PWD/.cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
```

## Lowering and ownership

The backend consumes the existing Pop DAG and reuses C expression generation,
view coordinate mapping, and lifetime-based arena planning. Reshape, permute,
shrink, flip, expand, load, and window remain indexing views. Canonical
multiply/reduce contractions lower directly to GPU matmul without allocating
broadcast products. Contractions with M/N >= 16 and K >= 32 use 16x16 output
tiles and 32-wide shared-memory operand tiles, coalescing B loads according to
its physical view layout. Smaller contractions with a contiguous B reduction axis
and K >= 32 use one warp per output; remaining cases use one thread per output.
Warp shuffles change reduction order; FMA stays disabled. Bounded, single-use
pointwise expressions fuse through views; shared tensors and contraction inputs
remain materialized to avoid recomputation. Shared scalar constant expressions
are inlined, and literal integer sequences such as `arange` become indexing
expressions without a device buffer. Materialized results write directly
to output buffers where possible. F32 reductions with at least 32 input values
use one warp per output when there are fewer than 256 outputs or the reduced
axis has contiguous storage. Other layouts and short/integer reductions keep
the sequential schedule. Sum/product reduction order changes, so numerical
comparisons use tolerances. Ordered max merges preserve the existing ternary
comparison behavior for NaNs and equal values, including signed zero.

The compiler recognizes row-local RMS normalization and stable softmax from
their primitive Pop graphs. For widths 32 through 4096, private reductions and
their pointwise consumers become one warp-per-row kernel. Softmax keeps each
lane's exponentials in registers for summation and output; no exponential tensor
is materialized. Shared intermediate results stay available to other consumers,
and fusion cannot move reads across writable-state stores. These schedules use
the existing operations and require no model-source changes.

Contractions absorb bounded pointwise epilogues, including residual additions,
broadcast bias and activation gates. The planner accepts internal shared uses
such as `x * sigmoid(x)` while retaining any intermediate exposed to another
consumer or output. It schedules the fused kernel at the final expression so
independent tensor operands are ready, and never delays the contraction across
a state write. Private stacks with two through four inputs can similarly write
directly into a store through indexing views. This requires immutable buffers
or already materialized snapshots at the expression's leaves; expressions
reading writable memory retain their separate stack buffer for overlap safety.
`PUPPYGRAD_CUDA_EXPRESSIONS=0` disables these epilogues, constant/range inlining
and stack/store fusion. Set it before compilation to compare schedules.

`PUPPYGRAD_CUDA_REDUCTIONS=0` restores sequential reductions and disables row
fusion. `PUPPYGRAD_CUDA_ROW_FUSION=0` keeps parallel reductions while disabling
normalization/softmax fusion. Set these before compilation.
`row_fusion_count()` and `parallel_reduction_count()` describe the selected
schedules; the latter counts standalone reductions, excluding fused kernels.
Warp shuffle participation follows the [NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

Supported output/input buffers are F32, I32, U8, and Bool. Weak scalars and
compile-time shapes follow existing language semantics. Integer add/subtract,
multiply, negation, and reductions explicitly wrap. FMA contraction and flush
to zero are disabled, with precise division/sqrt requested. CPU and GPU
transcendentals can still differ; floating-point parity uses tolerances.

Input slots, dtypes, and lengths are checked before GPU execution. Gather
bounds errors set a device error flag, prevent subsequent kernels from reading
invalid results, and become a host error. Kernels run in order on the default
stream of the retained primary context. The runtime pushes/pops that context
around execution and synchronizes before retained storage can be reused, including
on launch failures. Executables deliberately cannot be shared across Rust
threads implicitly.

Repeated executions use CUDA driver graph replay by default. An executable
records its lowered kernels as an ordered dependency chain and submits that
chain with one `cuGraphLaunch`. Kernel arithmetic and counts are unchanged;
input uploads, error-flag reset, synchronization and output downloads still
happen per call. This also applies to programs with retained writable state:
`STORE`/`AFTER` dependencies and arena reuse keep their existing execution order.

Each executable caches one recorded graph. Any allocation or replacement in the
shared runtime invalidates it, so changing input shapes, growing scratch/output
storage, or replacing a state declaration cannot replay stale addresses. New
input contents and state resets reuse the graph. Recorded graphs are destroyed
before unloading their kernel modules. GPU storage counters describe retained
tensor buffers and exclude driver-internal graph allocations.

Set `PUPPYGRAD_CUDA_GRAPH=0` before creating a runtime to use direct kernel
launches, or call `Runtime::set_graph_replay(false)`. `execution_stats()` exposes
graph builds, graph launches and direct kernel launches for the shared runtime;
`kernel_count()` continues to report the number of GPU kernels per execution.
The driver bindings use the original `cuGraphAddKernelNode` parameter ABI and
`cuGraphInstantiate_v2`; the driver copies kernel argument values when a node
is added ([NVIDIA graph API](https://docs.nvidia.com/cuda/archive/11.0/cuda-driver-api/group__CUDA__GRAPH.html)).

A `cuda::Runtime` owns resident inputs, writable state, an arena, output buffers,
and the device error flag. `compile` creates a private runtime; `compile_with_runtime` shares
residency among related executables. The LLM provider uses one runtime for all
static input-shape/capacity specializations, so weights and writable state are
shared rather than duplicated per shape. Scratch buffers grow when needed and reuse their capacity for smaller
shapes. Final outputs are copied into independent host tensors.

Inputs are immutable Arc-backed tensors. The runtime retains the previous host
Arc for each input slot and skips copies when dtype, length, and allocation
identity match. Replacing a tensor, including safe `Arc::make_mut` updates,
triggers an upload. Retaining the Arc prevents address-reuse false hits. A fresh
token tensor uploads only its IDs. Warm calls allocate no device memory; the
four-byte error flag is reset every invocation. Failed uploads invalidate their
cached identity; gather failures can recover on the next call.

Dropping an executable unloads its module. Dropping the final runtime/executable
owner frees all retained device buffers under their context, then releases the
primary-context reference. In the LLM provider, `free_model` drops every shape
specialization and its shared runtime. There is no global allocation cache that
keeps VRAM alive after the model is freed.

`Runtime::residency_stats` / `Executable::residency_stats` expose allocation,
input-upload count/bytes, and retained buffer capacities. Upload counters exclude
the error flag reset; resident bytes exclude driver/module memory. The planned
arena plus outputs per executable remains capped at 2 GiB; retained inputs and
the shared runtime's capacity across shapes are additional memory. Memory plans
respect producer/consumer lifetimes and duplicate outputs.

Generated `.cu` and `.ptx` files are cached under `.cache/pup/cuda/`, keyed by
source, compute capability, and NVRTC major/minor version. Runtime compilation
uses NVRTC even when retrieving a cached module to identify the compiler version.
Cache writes replace files atomically. No weights are embedded in emitted code.

## Current limits and next steps

The LLM `.pup` adapter accepts CUDA. Image generation and the generic training
harness still select CPU; CUDA profiling, standalone PTX emission, and GPU
provider-library export are not implemented yet. `emit --backend cuda --profile`
reports an explicit error. Existing CPU profiling remains available.

The backend uses shared-memory matmul tiling. Tensor cores and cuBLAS are not
used, and reductions have no warp-level parallelism. GPT-2 continues
to compile per static prefix length and has no retained GPU KV cache.
The next improvements are register tiling, parallel reductions, GPU event
profiling, then image/training integration.

## Verification

GPU-independent source/CLI tests run with the ordinary test suite. Explicit
hardware tests require a working GPU and NVRTC:

```sh
cargo test --test compiler_cuda -- --ignored --test-threads=1
```

They compare CPU/GPU matmul, batched strided contractions, normalization,
transcendentals, padded convolution, integer overflow, launch tails, empty
buffers, and duplicate outputs. They also check missing/wrong input buffers,
negative/high gather indices, repeated execution after errors, updated inputs,
and PTX cache reuse. Residency tests cover sharing across shapes, zero allocations
on warm runs, weight copy-on-write, independent host outputs, and releasing VRAM
while the primary context remains alive. Run the VRAM release check with
`cargo test --lib compiler::cuda::ownership_tests -- --ignored --test-threads=1`.
Real-model GPT-2 verification uses `--verify-reference`.

Validated on this workstation's RTX 2070 with NVRTC 12.6.85: the unchanged
`examples/llm.pup` generated eight GPT-2 tokens with a matching greedy choice
at every step. Maximum absolute logit error across those steps was 0.0002861.
The output was `The meaning of life is not the same as the meaning of death`.
This was a parity run with reference verification enabled, not a speed benchmark.
The local transcript is in `.cache/cuda-toolchain/gpt2-cuda.log`.

## Initial GPT-2 comparison with tinygrad on CUDA

On this RTX 2070, F32 full-prefix GPT-2 forwards (no KV cache) were compared
with tinygrad revision `1a58c3ae9d5ff5605d81085cf1a364c113e95cf6`, using
TinyJit and `BEAM=0`. Medians are from 11 alternating trials after 3 warmups.
Both return final logits to host; compilation, checkpoint loading, tokenization,
sampling, and reference verification are excluded.

| Prefix tokens | Puppygrad before retention | tinygrad resident weights | tinygrad reuploads weights |
|---:|---:|---:|---:|
| 5 | 57.77 ms | 10.45 ms | 60.09 ms |
| 12 | 58.93 ms | 11.31 ms | 64.82 ms |
| 32 | 66.27 ms | 18.56 ms | 75.68 ms |
| 128 | 99.58 ms | 40.09 ms | 97.68 ms |

In the original backend, normal tinygrad was 2.5–5.5x faster. Puppygrad uploaded
497,759,232 bytes of weights every invocation; separate profiling measured 47–48 ms for those
copies. At five tokens, this is about 80% of profiled wall time. The reupload
variant deliberately gives tinygrad the same weight-transfer work, but it still
retains GPU allocations; it is a diagnostic, not its normal inference path.

All tested prefixes matched greedy predictions; maximum absolute logit error
was 0.0002747. Tinygrad also generated exactly the same eight output IDs as the
Puppygrad CUDA parity run. Tinygrad uses default floating-point compilation;
Puppygrad disables FMA and preserves sequential reduction order.

Puppygrad launched 698 GPU kernels at five tokens, versus 174 CUDA math kernels
in tinygrad's captured schedule. These measurements describe the original CUDA
backend before residency, fusion, and shared-memory tiling. Planned arena size
is not a measure of total GPU residency.

The complete local report and raw samples are in
`.cache/cuda-compare/comparison.md` and `.cache/cuda-compare/results.json`.
Profiling was added only to an ignored snapshot of the compiler/runtime; it
asserts identical emitted CUDA source and outputs to production. The main
speed measurements use the unmodified production release-library executor.
GPU timings use [CUDA events](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
and are collected separately from uninstrumented wall-time comparisons.


## Resident and tiled GPT-2 results

The CUDA runtime now shares weights and scratch capacity across prefix
specializations. Single-use pointwise fusion preserves shared residual values,
contractions use shared-memory tiles when eligible, and materialized outputs
write directly to their destination. No `.pup` source changes were required.

A fresh paired run on the same RTX 2070 and pinned tinygrad revision used the
production release executor, 3 warmups, 11 alternating trials, and the same
full-prefix F32 GPT-2 workload (no KV cache, host logits, tinygrad `BEAM=0`).
Compilation/loading/sampling were excluded:

| Prefix tokens | Puppygrad | tinygrad resident | Puppygrad speedup |
|---:|---:|---:|---:|
| 5 | 5.07 ms | 9.79 ms | 1.93x |
| 12 | 6.36 ms | 11.34 ms | 1.78x |
| 32 | 9.79 ms | 18.42 ms | 1.88x |
| 128 | 27.92 ms | 39.86 ms | 1.43x |

Every warm invocation made **zero device allocations** and uploaded only token
IDs: 20, 48, 128, or 512 bytes (plus the four-byte error-flag reset). All
497,759,232 weight bytes were uploaded once, shared across all four prefix
variants. Kernel count fell to 288 per forward from 698 at five tokens / 821
at 128. Planned arena sizes are 138,284 / 331,932 / 885,772 / 3,948,556 bytes.
The shared runtime retains about 475–479 MiB including weights, outputs, and
scratch; this excludes module/driver memory.

Final logits are bit-for-bit identical to the previous Puppygrad CUDA output
at every tested prefix. Greedy predictions match tinygrad, with maximum absolute
logit error 0.0002747. The report and raw samples are in
`.cache/cuda-resident/comparison.md` and `.cache/cuda-resident/results.json`.
The benchmark asserts token-only uploads and zero allocations on warm calls.

The final LLM FFI provider also generated the same eight GPT-2 tokens with
reference verification enabled (maximum absolute error 0.0002861). Its transcript
is `.cache/cuda-resident/gpt2-parity.log`; that run includes compilation of
previously unseen prefix shapes and reference work, so it is not a warm-speed
measurement. The explicit cleanup test passed while retaining the primary
context externally, confirming device allocations are released at final-owner
cleanup rather than relying on context destruction.


## Retained state and small contractions

Generic `state`, STORE and AFTER operations retain writable GPU buffers across
input-shape specializations. Writes update indexed rows on-device; AFTER preserves
launch dependencies. Reset zeros state; final runtime/executable-owner cleanup
frees it. [Retained-state semantics](retained-state.md) describe checked bounds,
duplicate indices and overlapping copies.

Small F32 contractions with a contiguous reduction axis in the right operand
use one warp per output. Lanes read adjacent elements and reduce using shuffles;
larger eligible matrices retain shared-memory tiling. Selection is based on
shapes and views, with no model names or attention opcodes. Warp reduction changes
floating-point accumulation order, so numerical checks use tolerances.

[Cached Qwen3](qwen3.md) uses these facilities with its KV algorithm in `.pup`.
On the tested RTX 2070 it decoded in 11.14 ms/token versus 25.59 ms for native
tinygrad, with equal F32 weights/KV and matching greedy IDs. Results and the
tradeoff from reserved-capacity attention are documented in the Qwen guide.
