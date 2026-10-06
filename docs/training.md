# Tensor files and generic training

```sh
cargo build --release --offline
target/release/puppygrad train examples/mnist.pup --epochs 5 --threads 1
target/release/puppygrad train examples/mnist_autodiff.pup --epochs 5 --threads 1 --cpu-target native
```

This command uses a generic Rust host and the C CPU compiler. There is no Python
trainer, MNIST runtime, autodiff dependency, or profiling syntax in the language.
The `.pup` program computes the forward pass, loss, explicit gradients, and SGD
update. The host handles loading, batches, initialization, state ownership,
evaluation, checkpoints, and metrics.

MNIST uses the original 60,000 training / 10,000 test split, downloaded on demand
from the [authorized CVDF mirror](https://github.com/cvdfoundation/mnist). Archive
MD5 checksums are checked against the original files (also listed by
[torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)).
Downloads and results are under ignored `.cache/`. Archive verification uses
`md5sum`; generated CPU code uses `cc`, libc, libm and pthreads.

## Files become tensors

`runtime::data::load(path, options)` returns named tensors containing dtype,
shape, and contiguous little-endian element bytes. It does not normalize,
flatten, one-hot encode, or silently cast data.

- **IDX:** reads rank, dimensions and dtype from the header, validates exact
  payload size and converts multibyte elements from big endian. The sole tensor
  is named `tensor`. All IDX numeric types are readable.
- **CSV:** requires an explicit numeric column schema. Fields can be grouped into
  fixed-size tensor rows. Quoted headers/fields and CRLF are supported. Missing,
  malformed, non-finite and out-of-range numeric values fail with a row/field
  error; no string vocabulary or imputation is inferred.
- **Safetensors:** exposes each stored tensor by name, preserving its shape and
  dtype. Supported file dtypes are U8, I8, I16, I32, F32, F64 and BOOL; other types
  produce an explicit error.
- **Gzip:** decompression is a separate layer, detected by magic bytes. `.gz`
  requires a valid gzip header. Decompressed files are limited to 1 GiB.

Extensions select a format; parsers validate content. IDX magic also recognizes
original names such as `train-images-idx3-ubyte.gz`. A `format` override supports
misleading/unknown extensions. Unknown formats fail rather than being guessed
as CSV. Multi-tensor files require selecting a tensor by name.

The compute compiler currently accepts `u8`, `i32`, `f32`, and `bool`. Loading a
wider file dtype is supported, but binding it to a program fails clearly until
that compute dtype is implemented. Byte tensors are genuinely U8 storage, not
boolean buffers or implicit float conversions.

MNIST's images remain `u8[60000,28,28]` and labels `u8[60000]`. Batches preserve
those sample dimensions. `mnist.pup` performs casts, pixel normalization,
flattening, and one-hot encoding using ordinary operations.

## Host contract

The default sidecar is `SOURCE.train.json`; `--config PATH` overrides it. This
is host configuration, not `.pup` syntax. See [mnist.train.json](../examples/mnist.train.json).

- `version: 1`, `batch_size`: static batch specialization.
- `datasets`: named inputs; each has `train` and `test` file descriptions.
- File description: `path`, optional `format`, `tensor`, `csv`, and optional
  `url`/`md5` for verified downloads. Paths are relative to the configuration.
- `state`: any number of F32 state tensors with input name, exported output name,
  shape, and initialization (`zeros`, `normal` with `stddev`, or `constant` with
  `value`). This can represent parameters and optimizer state.
- `learning_rate`, `valid`: names of host-supplied F32 scalar and row-mask inputs.
- `loss`: exported F32 scalar, defined by the program as mean loss over valid rows.
- Optional `classification`: exported `[batch,classes]` prediction binding and a
  dataset tensor of scalar U8/I32 class labels. Omit for regression/other losses.

The source binds names using existing `input("name")` calls. The host supplies
shapes/dtypes in the compiler context. No parameter count, image shape, class
count or MNIST-specific logic is built into `train.rs`.

Every dataset tensor in a split must have the same nonzero number of rows;
train/test dtypes and per-row shapes must agree. State/output mappings and metric
shapes are validated before training. Unsupported/different contracts fail.

Rows are shuffled together, batched on dimension zero, and the final batch is
zero-padded. `valid` is 1 for actual rows and 0 for padding. The program must use
that mask in loss and gradients. Updated state is retained only after a training
step. Evaluation supplies learning_rate=0 and discards updated state; the program
must report pre-update predictions/loss. Evaluation executes the training graph,
including its backward work, so these are not inference-only timings.

Example numeric CSV schema in a file description:

```json
{
  "path": "samples.csv",
  "tensor": "features",
  "csv": {
    "columns": [
      {"name": "features", "fields": ["x", "y"], "dtype": "f32", "shape": [2]},
      {"name": "target", "fields": ["target"], "dtype": "i32"}
    ]
  }
}
```

The integration suite trains scalar regression directly from CSV with a different
batch size and a single scalar state, independent of the MNIST experiment.

## Metrics and checkpoints

Each run creates a new output directory; `--output-dir` selects it. Existing
results are never overwritten. Outputs include:

- `metrics.json`: epoch loss/optional accuracy, row throughput, train-step
  median/p95, compiler/cache timing, host/compiler settings and detailed counters.
  Build settings include CPU target, compiler identity, effective flags and the
  native CPU fingerprint when applicable.
- `compiler-metadata.json`: kernel IDs, root operation, dtype, shape, matching
  top-level source bindings/lines, packed bytes and estimated matmul FLOPs.
- `kernels.csv`: accumulated kernel, packing and compute durations.
- `state.safetensors`: final named state tensors.
- `program.pup`, `program.train.json`, `program.c`: source, resolved host contract,
  and emitted C snapshot. The current CLI initializes a new run; checkpoint
  resume is not implemented yet.

Instrumentation belongs to the compiler/runtime. `cpu::compile_profiled` and
`Executable::run_profiled` work independently of the training harness. For a
self-contained source program, `puppygrad emit examples/matmul.pup --profile -o
matmul.c` exposes the same optional ABI, documented in
[puppygrad_profile.h](../include/puppygrad_profile.h). Programs with named external
inputs need a compiler `Context` as supplied by the training host.

Counters reset per invocation into caller-owned storage; there are no mutable
profiling globals. Normal `cpu::emit` / `compile` omit instrumentation. Source
metadata is separate from graph identity. This first version labels kernel roots
using top-level bindings; it does not yet reconstruct full nested source call
paths or assign fictitious timings to each operation inside a fused kernel.

Training counters exclude dataset loading, compilation, warmup and evaluation.
Kernel elapsed includes packing and matmul compute/synchronization; those nested
fields must not be added again. Packed bytes count the payload written into
packing buffers, not measured memory-bus traffic. Matmul FLOPs are static
estimates. Workspace is the generated C tensor arena, excluding input/output
buffers, host dataset storage and thread stacks. Timings include instrumentation
overhead; host-call timings additionally include output allocation and FFI.

The CPU compiler now reuses intermediate storage by lifetime, fuses bounded
pointwise expressions and matmul epilogues, and writes eligible outputs directly
to caller buffers. The separate full-operand packing phase is eliminated, so its packing
counters are zero. Fixed tile panels are populated inside compute time. The arithmetic still runs entirely in generated C.

`memory_plan` records each internal allocation's bytes, arena offset, first and
last kernel (inclusive), plus `peak_live_bytes`. A last-kernel index equal to the
kernel count means the buffer survives through final output copying.
`matmul_stack_scratch_bytes_per_worker` separately reports the fixed declared
float arrays used by a matmul worker; it is not a measurement of the entire C
stack or process RSS. No full tensors are moved to hidden stack allocations.

## Autodiff example

The handwritten derivative baseline remains `examples/mnist.pup`. To train the
same model with compiler-generated gradients:

```sh
cargo run --release --offline -- train examples/mnist_autodiff.pup --epochs 5 --threads 1
```

Its `grad(loss, w1)` calls expand into the same Pop vocabulary used by the
forward model. The training contract, dataset loaders, explicit SGD updates and
telemetry work unchanged. See [autodiff.md](autodiff.md) for the initial scope.
