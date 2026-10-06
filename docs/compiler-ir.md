# Pops are the .pup language model

A `.pup` program constructs a DAG of Pops. A binding names a value; it does not
allocate a tensor or execute a kernel. Shapes are values in the same graph.
Matmul, convolution, attention, and normalization are intended to be library
compositions of primitives, not mandatory compiler opcodes.

The compatibility target is tinygrad's UOp specification at revision
`1a58c3ae9d5ff5605d81085cf1a364c113e95cf6`:

- [Stage specifications](https://github.com/tinygrad/tinygrad/blob/1a58c3ae9d5ff5605d81085cf1a364c113e95cf6/tinygrad/uop/spec.py)
- [Operation vocabulary](https://github.com/tinygrad/tinygrad/blob/1a58c3ae9d5ff5605d81085cf1a364c113e95cf6/tinygrad/uop/__init__.py)
- [Node, dtype inference, shape inference, and movement semantics](https://github.com/tinygrad/tinygrad/blob/1a58c3ae9d5ff5605d81085cf1a364c113e95cf6/tinygrad/uop/ops.py)
- [Dtypes and constant representation](https://github.com/tinygrad/tinygrad/blob/1a58c3ae9d5ff5605d81085cf1a364c113e95cf6/tinygrad/dtype.py)

This is an initial static subset, not full tinygrad compatibility. Unsupported
operations and symbolic shapes fail explicitly. There is no Python runtime
dependency for parsing or checking `.pup` programs.

## Try it

```sh
cargo run -- check examples/matmul.pup --dump-pops
```

The example describes `[2,3] @ [3,4]` using parameters, shape constants, views,
multiply, and reduction. Its result has shape `[2,4]`. The multiply's `[2,4,3]`
shape is a logical iteration domain, not an allocated intermediate.

```text
a = param(0, f32, 6)
b = param(1, f32, 12)
left = reshape(a, [2, 1, 3])
matrix = reshape(b, [3, 4])
transposed = permute(matrix, [1, 0])
right = reshape(transposed, [1, 4, 3])
products = mul(left, right)
contraction = permute(products, [2, 0, 1])
result = reduce(contraction, add, 1)
output result
```

Run the complete GPT-2 model through the C CPU backend (requires `cc` and a GPT-2
checkpoint containing `config.json`, `tokenizer.json`, and `model.safetensors`):

```sh
cargo run --release -- run examples/llm.pup --device cpu \
  --model-dir models/gpt2 --prompt "The meaning of life is" \
  --max-new-tokens 8 --verify-reference
```

`run` (also available as `llm`) uses the [LLM runtime and FFI contract](llm-runtime.md).
It supports streaming (`--stream`) and sampling (`--temperature`, `--seed`). A provider
adapts the current `.pup` graph and GPT-2 checkpoint to that contract. The `.pup` file
defines the model arithmetic. `--verify-reference` independently
compares every step's logits and greedy token choice with the existing Rust GPT-2 model
(max absolute logit error limit: 0.003). It is optional and loads a second copy of
the weights. `check` remains allocation-free and needs no model assets; programs
using `input`, `weight`, or `config` require a binding context, supplied by `run`.
An initial [CUDA backend](cuda-backend.md) runs the same GPT-2 source through
NVRTC and the NVIDIA driver with `--device cuda:0`. CUDA emission is available
without a GPU using `emit --backend cuda`. Standalone PTX, OpenCL, and SASS
backends remain future work; unsupported devices fail explicitly.

## Initial source syntax

Expressions support nested calls, parentheses, lists, numeric literals, unary
`+`/`-`, and infix `+`, `-`, `*`, `/`, `<`, and `>`. Multiplication/division bind
more tightly than addition/subtraction. `//` and `==` currently require static
integer operands; static integer arithmetic is checked for overflow and division
by zero. `/` converts integer operands to f32. Expressions occupy one line.

Names are ASCII identifiers and sources must be declared first. Bindings are
immutable except inside bounded graph-construction loops. `#` starts a comment
outside string literals. `output expression, ...` declarations come last and
construct a `SINK`. Primitive operation names are case-insensitive.

## Functions and the source library

Functions use four-space indentation, local immutable bindings, and a final
`return` expression. Parameters are graph values. Calls expand into
primitive Pops at check time; no new `CALL` operation or execution mechanism is
introduced. Definitions may call functions declared later in the file.

```text
def projection(x, weight, bias):
    result = linear(x, weight, bias)
    return result

x_storage = param(0, f32, 6)
weight_storage = param(1, f32, 12)
bias = param(2, f32, 4)
x = reshape(x_storage, [2, 3])
weight = reshape(weight_storage, [4, 3])
y = projection(x, weight, bias)
output y
```

The bundled [nn.pup](../stdlib/nn.pup) source is automatically in scope in every
program. Call `matmul`, `linear`, `layer_norm`, and the other functions directly;
no import or namespace prefix is needed. The source is embedded into the compiler,
so checking works independently of the working directory. It provides:

- `matmul(a, b)`: rank-two `[M,K]` and `[K,N]` inputs to `[M,N]`.
- `linear(x, weight, bias)`: `[M,K]` inputs, `[N,K]` weights, `[N]` bias to
  `[M,N]`. Rank and contraction/bias dimensions are checked explicitly.
- `batched_matmul`, `dense` (checkpoint weights in `[input,output]` order),
  `layer_norm`, `exp`, `gelu`, `softmax`, and `causal_attention`.
  These are ordinary `.pup` functions, not compiler-recognized model operations.

Functions can call other functions but cannot capture module values or caller
locals; pass all data explicitly. Names are case-sensitive and cannot override
primitive or standard-library function names. A call must supply exactly one value
per parameter. Repeated calls with the same argument values reuse their expansion. Recursive calls are
rejected and expansion depth is limited to 128. Function declarations and body
syntax are checked when parsed; operation types and referenced values inside
bodies are checked when called. Errors include call-site and body line context.

Two frontend helpers support static shape-specialized definitions:

- `n = dim(x, 1)` emits an integer `CONST` for a known input dimension. The axis
  is a static integer expression and out-of-range axes fail checking.
- `assert_eq(a, b)` checks two integer constant values at expansion time and
  emits no operation. It is useful for contraction constraints, including empty
  tensors where reshape alone cannot detect a mismatch.

Shape lists may mix literals and local dimension values, for example `[m, 1, k]`.
They still lower to `CONST`/`STACK` shape values. Legacy `import nn` and `nn.function(...)` spellings remain accepted and
resolve to the same functions. Filesystem imports, runtime recursion, closures,
and attribute parameters are not implemented.

`for layer in range(count):` expands a bounded static loop (0–4096 iterations).
Reassignments create new graph values; pre-existing bindings carry out of the loop,
while loop-local names do not escape. Total expansion is capped at 100,000
statements. This is compile-time graph construction, not a runtime loop instruction.

The host can bind tensors with `input("tokens")` and `weight("wte.weight")`, and
numeric constants with `config("n_layer")`. These become PARAM/view/CONST nodes.
Restricted f-strings interpolate integer binding names, for example
`weight(f"h.{layer}.ln_1.weight")`. `arange(n)` constructs a static i32 sequence
(up to 65,536 elements). See [llm.pup](../examples/llm.pup) for a complete model.

## Primitive operations

| Operation | Arguments and semantics |
| --- | --- |
| `param(slot, dtype, size)` | External flat tensor; use `scalar` instead of size for a scalar parameter. Slots are unique within a program. |
| `const(value)` | Bool, signed i64 integer literal, or finite f64 float literal. Integers/floats have weak dtypes; use `cast` to state a concrete width. |
| `stack(a, b, ...)` | Stack equal-shaped values; also builds shape tuples. `stack()` is the empty shape tuple. |
| `cast(x, dtype)` | State a concrete `bool`, `i32`, or `f32` dtype. |
| `add`, `sub`, `fdiv`, `mul`, `max`, `cmplt` | Two sources with compatible dtypes; shapes broadcast from the right. `cmplt` produces bool. |
| `neg`, `exp2`, `log2`, `sqrt`, `sin` | One source. The transcendental subset requires floating-point input. |
| `where(condition, a, b)` | Bool condition, compatible branch types, broadcast shapes. |
| `index(base, indices)` | Integer gather along the first axis; result shape is `indices.shape + base.shape[1:]`. |
| `load(indexed)` | Read an INDEX value. External inputs remain read-only; state reads can be ordered with AFTER. |
| `state(name, dtype, shape)` | Named zero-initialized writable buffer retained across executions; lowers to a writable PARAM. |
| `store(index(state, rows), value)` | Update matching rows; returns a statement. Stores have distinct identity and checked bounds. |
| `after(value, writes...)` | Preserve a tensor/view and add dependencies on completed writes. |
| `reshape(x, shape)` | Preserve element count. |
| `expand(x, shape)` | Prepend axes: input `[2,3]` with shape `[4]` becomes `[4,2,3]`. |
| `permute(x, [axes...])` | Reorder all axes. |
| `pad(x, offsets, sizes)` | Place input at offsets inside the output sizes. Sizes are total output dimensions, not right-padding counts. |
| `window(x, [kh, kw], [sh, sw])` | NCHW sliding-window view, shaped `[N,C,OH,OW,KH,KW]`; positive kernel/stride sizes and no implicit padding. This is a Puppygrad view extension used to lower convolution without patch storage. |
| `shrink(x, offsets, sizes)` | Slice at offsets for the given output sizes. |
| `flip(x, [flags...])` | One bool flag per axis. |
| `reduce(x, op, count)` | Reduce `count` leading axes with `add`, `mul`, or `max`. Arbitrary-axis reduction is a permutation followed by this operation. |
| `sink(a, b, ...)` | Collect graph roots; does not produce a tensor. |

Shape arguments to reshape/expand/pad/shrink can name graph values. A list such as
`[2,3]` is shorthand for integer `CONST`s and a `STACK`. A single dimension uses
a scalar `CONST`; an empty shape uses an empty `STACK`. These are ordinary graph
nodes, not hidden shape fields on operations.

## Graph contract

- A node is `(op, src, arg)`. Dtype and shape are derived. Source names and line
  numbers live separately so equivalent expressions still share a node.
- Nodes are immutable and structurally interned within an arena. The arena owns
  them until dropped; values cannot be used in a different graph.
- Arity, argument kinds, dtypes, broadcast compatibility, permutation validity,
  shape products, and movement bounds are checked at construction.
- Floating-point constants are keyed by their IEEE bits, preserving signed zero
  and NaN payloads. The source parser currently accepts finite literals only.
- An iterative topological walk visits reachable nodes once. Rewrites rebuild
  sources bottom-up, preserve shared subexpressions, and check replacement types
  and shapes. Each pass visits the original graph once; no implicit fixed point.
- Initial simplification removes identity views/casts. It does not assume that
  floating-point `x + 0` or `x * 0` preserves all IEEE behavior.
- STORE statements have unique identity and are never structurally merged.
  AFTER adds explicit ordering dependencies while preserving storage identity.
  [Retained-state semantics](retained-state.md) cover ownership, writes, reset,
  aliasing and input-shape sharing. Other effectful operations need their own
  identity and ordering contracts.

## Stage boundaries and remaining work

The target is the full stage-aware Pop architecture, not a single permissive
validator. A node being legal in one stage does not imply it is legal in another.

| Upstream spec | Current implementation | Next work |
| --- | --- | --- |
| `spec_shared` | Constants, parameters, stack, casts, selected ALU, sink, indexed state STORE and AFTER | Remaining arithmetic, ranges, general buffer stores, calls, hardware operations |
| `spec_tensor` | Six movement ops, static shapes, leading-axis reductions | Symbolic shapes, full dtype promotion, remaining tensor and device operations |
| `spec_program` | Checks stage restrictions on the implemented subset: no movement/reduce, concrete widths except CONST, CONST sources under CAST | Range/index lowering, loads/stores, barriers, legalization, code generation |
| `spec_kernel_graph` | Checks legal parameters/constants/stack/cast/sink in the subset; rejects tensor ALU | Buffers, calls, effect dependencies, scheduling and memory planning |
| `spec_hcq` | Not implemented | Device command queues and address resolution |
| `spec_full` | Not implemented | Explicit verification rules for intermediate rewrite stages |

The current `ParamArg` supports slot, dtype, and optional flat size. It does not
yet implement tinygrad's ranges, address spaces, device tuples, volatile storage,
images, or owned buffers. Shape expressions must be nonnegative i64 constants or
stacks of them; symbolic expression evaluation is deliberately unsupported.
Integer constants are bounded by i64 rather than Python's arbitrary precision.

The intended pipeline is `.pup -> tensor Pops -> indexed/scheduled Pops -> backend
code + execution plan`. Backend-specific operations such as `WMMA` and `INS`
belong in later lowering stages. A tensor multiply/reduce pattern can become a
hardware matrix operation without introducing `MATMUL` into the language core.
Backend renderers, compilers, and executors remain separate boundaries.

The first executable backend lowers tensor Pops directly to C kernels with explicit
index expressions for views. It does not yet construct a separate scheduled Pop
IR. Canonical multiply/permute/leading-add-reduce contractions lower to tiled C
matrix kernels included in the emitted source. Operands are read through their
view expressions using bounded tile scratch; the broadcast product and full
packed matrices are never allocated. Other operations use generated C
loops. There are no Rust compute callbacks, BLAS dependencies, or host function
pointers in the generated library. Rust remains the compiler and optional launcher.

The internal C ABI is `int pup_run(const void **inputs, void **outputs, size_t threads)`.
The caller supplies validated input/output buffers and a positive thread limit.
Return codes are 0 (success), 1 (allocation failure), 2 (gather bounds violation),
and 3 (invalid thread count or worker initialization failure). Emitted source can
be built with `cc -std=c11 -O2 -fno-math-errno -fwrapv -ffp-contract=off -pthread`
and linked with `-lm`, including from a standalone C application. Standard libc,
libm, and POSIX threads are the only runtime dependencies.

Matrix output tiles (4 rows by 8 or 16 columns, with tails) are distributed across a C
pthread pool. The calling thread participates; the pool is created once per
`pup_run`, reused across contractions, and joined on success and error exits.
Weight panel packing is serial; large independent elementwise, padding, and
reduction output loops use the same pool. Small loops remain serial. Each dot product retains its
reduction order across thread counts. This is an initial C implementation, not a
replacement for the removed tuned GEMM library's performance claims.
Generated source and shared libraries are cached under ignored `.cache/pup/cpu/`.
The cache key includes C source, CPU target, compiler path/identity, target triple
and flags. Native targets additionally fingerprint effective compiler target
macros and host CPU features. Each entry has a validated build manifest; old
entries without one are rebuilt. Each shape specialization compiles separately.

`train`, `run`, `llm`, and `llm benchmark` accept
`--cpu-target generic|native|avx2`. `generic` preserves the compiler's default
instruction target; `native` adds `-march=native`; `avx2` adds `-mavx2` and requires
an AVX2-capable x86 host. Generated C uses GCC/Clang vector types for register microkernels and automatic
vectorisation for other loops. The optimization level remains `-O2` and `-ffp-contract=off` remains set.
No SIMD syntax is needed in `.pup`. `emit` produces the same portable C source;
choose target flags when compiling it externally. Already compiled LLM shared
libraries cannot be retargeted by this switch.

Rust hosts can use `cpu::compile_with_options` or
`cpu::compile_profiled_with_options` with `cpu::BuildOptions { cpu_target: ... }`.
Existing compilation APIs default to `generic`. `Executable::build_info` reports
the actual compiler and flags, also included in profiled executable metadata.
See [CPU targets and local measurements](cpu-targets.md).

Execution validates input slots, dtypes, and lengths before entering generated
code. Gather indices are bounds-checked. Allocation failures propagate as errors;
planned arena storage, bounded weight panels, and outputs are limited to 2 GiB.
Intermediate buffers share one workspace allocation. A deterministic lifetime
planner assigns aligned offsets,
reusing regions only after the final consuming kernel completes. Views retain
their underlying storage; returned views remain live through output copying.
Eligible concrete outputs are written directly to disjoint caller-owned buffers.
Output dtypes must be concrete; failed calls may leave outputs partially written.

Bounded elementwise expressions are fused into their consumers, retaining dtype
rounding at each operation. Single-consumer pointwise matmul epilogues (such as
bias or SGD updates) are evaluated before storing each output tile. Matmul reads
strided operands through views, using direct reads for small contractions and a
bounded packed weight storage for eligible larger contractions. No full operand or
convolution patch matrix is allocated. Coordinates stay separate through matching
reshape groups, avoiding repeated flattening and decoding of independent indices.
Each worker uses 4-by-8 or 4-by-16 register accumulators and at most 320 bytes of
declared float array scratch. Packed storage is refreshed per invocation, capped
at 512 KiB across workers, and reported separately as `packed_workspace_bytes`;
mutable parameters
never rely on a pointer-based packing cache. No matrix-sized allocation is hidden
on the stack. Large strided contractions use private 512-by-16 cache panels and
128-by-16 output macro tiles. Each worker packs and computes its own chunks without
intermediate pool barriers; partial sums occupy the existing destination buffer.
Fused epilogues execute only after the final chunk. Smaller packed contractions
use a shared serially packed panel. Kernel completion remains synchronous at the
memory-reuse boundary. Profile metadata labels overlapping packing/compute phase
counters as `worker_average`; whole-kernel elapsed time remains wall time.

Current limits: CPU shared-library execution, static shapes, and
full-prefix recomputation for each generated token. No KV cache or GPU execution yet. GELU uses the
usual tanh approximation expressed through exp2 primitives. The Rust matmul helper
remains a decomposition example; `matmul` is implemented in source.

## Conformance evidence

`tests/data/compiler/tinygrad-spec.json` records types, shapes, rejected nodes, and
program/kernel-graph acceptance from the pinned upstream implementation. Normal
Rust tests consume this checked-in corpus without importing tinygrad. Independent
scalar-index tests also compare the matmul graph against ordinary dot products.

Regenerate the corpus with an unmodified checkout of the exact revision:

```sh
PYTHONDONTWRITEBYTECODE=1 CACHELEVEL=0 python python/tinygrad_spec_fixtures.py \
  --tinygrad /path/to/tinygrad
cargo test compiler::
cargo test --test compiler_cli
```

The generator checks revision and source modifications before loading the oracle.
This corpus demonstrates the tested subset; it is not proof of full compatibility.
Changing the target revision requires reviewing semantic changes and regenerating
the fixtures deliberately.

`/third_party/` is ignored local reference material. The source links, revision,
fixture data, and generator are tracked. tinygrad's MIT notice is retained in
[licenses/tinygrad.txt](licenses/tinygrad.txt).

## Emit C without compiling or executing

```sh
puppygrad emit ./llm.pup
puppygrad emit examples/llm.pup --sequence-length 5 -o llm.c
puppygrad emit examples/matmul.pup > matmul.c
```

`emit` writes only self-contained C, to stdout by default (`-o -` is equivalent).
`-o FILE` writes it to a file. It never invokes `cc`, loads generated libraries,
runs model inference, or creates compiler cache files. Errors are reported on
stderr, and existing output files are untouched if generation fails.

Output is pretty printed with `clang-format` by default: four-space indentation,
100-column wrapping, and expanded function bodies and loops. Install
`clang-format` on PATH, or use `--raw` for compact output without a formatter.
The style is fixed, independent of local `.clang-format` files. Formatting only
affects `emit`; runtime compilation and its cache are unchanged. Formatter errors
also leave existing output files untouched.

Self-contained programs need no model assets. For `weight()` and `config()`
bindings, `--model-dir DIR` selects checkpoint metadata; otherwise emission uses
`models/gpt2` when external bindings are needed. It reads `config.json` and only
the `model.safetensors` header, validating the declared payload size without
reading weight values. No tokenizer or actual input token IDs are needed.
Bindings use the same sorted slots and normalized names as the LLM provider.
Weights remain caller-supplied F32 buffers, including when the checkpoint stores
F16/BF16. Generated comments identify the used input slots, types, and lengths.

The backend currently specializes static shapes. `--sequence-length N` sets the
`tokens` input shape to `[N]` (default 1); emit separately for each required length.
The result is the tensor program's `pup_run` entry point, not the LLM application's
tokenizer, sampling loop, or `get_llm_api` adapter.


## Tensor files and profiling

The CPU subset additionally supports concrete `u8` inputs, casts and outputs.
IDX/CSV/safetensors file readers and the generic training host bind named tensor
inputs without model-specific parsing or new profiling syntax. See
[training and tensor data](training.md).

`emit --profile` opts into caller-owned timing counters and a JSON metadata ABI.
Normal emission has no timers. `cpu::compile_profiled` exposes the same facility
to Rust hosts. Named-input training programs are specialized using the contract's
compiler context; their generated C is saved alongside training results.

## Reverse-mode differentiation

`grad(loss, value)` expands at compilation time into ordinary Pops. `loss` must
be a scalar f32 and `value` must be f32; the result has `value`'s shape. The
compiler accumulates shared-path contributions and reverses broadcasting and
views. Canonical matrix contractions have contraction-shaped pullbacks, without
adding a matmul or gradient opcode. There is no runtime tape or implicit state.
See [autodiff.md](autodiff.md) for supported rules, nonsmooth conventions and the
MNIST example. Indexed-input gradients currently require unimplemented scatter-add
and produce a compile error on requested differentiation paths.
