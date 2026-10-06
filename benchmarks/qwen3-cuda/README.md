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

To isolate graph replay from kernel arithmetic changes, after rebuilding:

```bash
python benchmarks/qwen3-cuda/replay.py --output-dir .cache/qwen3-replay
```

This compares two production FFI workers, one with direct launches and one with
graph replay, in shuffled paired trials at 24 and 128 input tokens. It requires
enough GPU memory for two resident F32 models (about 5 GiB total plus driver and
desktop usage). It checks bitwise forward-logit parity at 24, 64 and 128 tokens,
greedy token parity, warm graph reuse, and submission counters. Loading,
compilation and three warmups are excluded. Increase `--tokens` to measure a
longer decode. `benchmark.py` still compares the default replay path against
native tinygrad; set `PUPPYGRAD_CUDA_GRAPH=0` to benchmark the direct path.

The same paired FFI harness can isolate reduction scheduling and row fusion:

```bash
python benchmarks/qwen3-cuda/replay.py --compare reductions \
  --output-dir .cache/qwen3-reductions
python benchmarks/qwen3-cuda/replay.py --compare row-fusion \
  --output-dir .cache/qwen3-row-fusion
```

Both variants use graph replay. `reductions` compares the previous sequential
reductions with parallel reductions plus row fusion; `row-fusion` compares
parallel reductions with and without row fusion. Forward logits use a 0.003
maximum absolute-error limit and must keep the same argmax; all greedy output
tokens must match. Results include fusion counts, standalone parallel reduction
counts, warm one-token execution counters, and paired latency samples. Generation
can end early at EOS; JSON records the actual output-token count.

To isolate contraction epilogues, constant/range inlining and stack/store fusion:

```bash
python benchmarks/qwen3-cuda/replay.py --compare expressions \
  --output-dir .cache/qwen3-expressions
```

Both workers use parallel reductions, row fusion and graph replay. Only
`PUPPYGRAD_CUDA_EXPRESSIONS` differs. The same forward-logit and greedy-token
checks apply, and the warm decode profile records the resulting kernel count.

To measure larger KV capacities, prefill/decode time and memory independently:

```bash
python benchmarks/qwen3-cuda/context.py --output-dir .cache/qwen3-context
```

The defaults test 512, 2048, 4096 and 8192 capacity, using repeated chat tokens
and 128-token prefill chunks. Each prefix leaves 128 slots for decoding. Three
prefill samples and eleven one-token decode samples follow three warmups;
loading, compilation and state reset are excluded from runtime timings. The
prefill wall timer additionally includes benchmark orchestration and reset.
Results record early and late decode, kernel counts, runtime-owned GPU buffers,
driver-reported process GPU memory, current host RSS and host high-water RSS.
Each capacity gets a fresh runtime so larger retained allocations cannot inflate
the next row. Decode must allocate no buffers, rebuild no graphs and upload
only four token bytes. Repeated prefill after reset must reproduce bitwise logits.

The longest capacity also compares 128-token and 256-token prefill chunks using
full-logit tolerance 0.003 and the same argmax. A separate runtime executes the
largest single-shot prefill that passed metadata-only compiler planning, checks
it against chunked logits and records its much larger arena. Single-shot plans
at 16K and 32K report workspace failures without allocating GPU memory.

The capacity sweep orchestrates the retained `.pup` program directly, with a
fixed benchmark capacity. Production CUDA LLM FFI now also chunks prefill by
default and selects its effective context through the generic allocation planner;
`llm capacity` inspects that estimate. The fixed-capacity benchmark does not use
the automatic sizing policy, so each row has a controlled KV allocation.
Synthetic repetition and chunk invariance establish execution/memory behavior,
not long-range retrieval or model quality. These measurements use F32 weights
and F32 KV; the model's advertised context window does not imply it fits in GPU
memory at this precision. The optional fifth argument to the Rust forward
runner sets KV capacity; `plan CAPACITY TOKENS` performs driver-free lowering.
