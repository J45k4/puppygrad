# Qwen3 in Puppygrad

[examples/qwen3_cached.pup](../examples/qwen3_cached.pup) implements the dense Qwen3 decoder
using ordinary source functions and primitive tensor Pops. The same source supports
Qwen3-0.6B and Qwen3-1.7B, using dimensions from the checkpoint config. It runs through the
existing `llm` application and LLM buffer/FFI contract, on generated C CPU kernels
or generated CUDA/HIP kernels. There is no Qwen model computation in Rust or Python.

The source includes RMSNorm, per-head query/key normalization, half-split RoPE,
grouped causal attention, SiLU gated feed-forward blocks and tied output weights.
The bundled [source library](../stdlib/nn.pup) provides `rms_norm`,
`rotary_positions`, and `grouped_query_attention`; these expand into existing
operations and introduce no language syntax or model-specific compiler lowering.
Qwen3-0.6B has 16 query heads and 8 KV heads, each 128 wide. Its 1024-wide hidden
state does **not** imply a 64-wide attention head.

## Assets and execution

The catalog includes these official Apache-2.0 checkpoints:

| Model | Pinned revision | Native BF16 weights | F32 comparison |
|---|---|---:|---:|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | 1.11 GiB | 2.22 GiB |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 3.20 GiB | 6.41 GiB |

Their licenses are retained alongside downloaded models. Assets, generated code, PTX and benchmark logs
remain under ignored `models/` and `.cache/` directories.

The checkpoint loader keeps the original BF16 bits in its final shared buffers,
and uploads those same two-byte values directly to CUDA/HIP. F32 activations and
accumulators consume BF16 weights inside the kernels; this avoids a full widened
copy while keeping the previous model arithmetic. This path uses the existing
matrix schedules with F32 arithmetic, rather than hardware BF16 matrix instructions.

On 2026-10-08, release measurements on this RX 9070 XT with 512 KV slots,
a 33-token prompt and 64 output tokens gave these medians after one warmup and
three measured requests:

| Model / weight storage | Weight memory | First token | Decode tokens/s |
|---|---:|---:|---:|
| Qwen3-0.6B / previous F32 | 2.22 GiB | 72.7 ms | 125.0 |
| Qwen3-0.6B / native BF16 | 1.11 GiB | 66.5 ms | 156.7 |
| Qwen3-1.7B / native BF16 | 3.20 GiB | 143.3 ms | 88.1 |

The 0.6B comparison produced identical greedy token IDs, with approximately 25%
faster decode. Timings include production FFI reset, prefill, host sampling and
callbacks, and exclude loading and compilation. Activations, accumulation and
KV storage were F32 for every row. These short-context results do not predict
throughput at longer contexts.

Three alternating Qwen3-1.7B checkpoint-only loads measured a median of 2.074 s
with the previous optimized F32 loader and 0.777 s with native BF16. Median peak
process RSS fell from 6571.7 MiB to 3286.8 MiB. The OS file cache was not cleared;
these are mostly warm-cache measurements and exclude GPU upload, tokenization,
planning and compilation. Detailed samples and token IDs are recorded in
`.cache/bf16-validation/results.json`.

Before native BF16 storage was added, a local Qwen3-1.7B checkpoint-only
measurement on 2026-10-08 (same files, before and after removing intermediate F32
weight copies) showed:

| Build | Before | After |
|---|---:|---:|
| Development (`cargo run`) | 39.34 s | 6.49 s |
| Release | 8.86 s | 4.88 s |

These are individual runs without clearing the OS file cache; disk-cache state
and memory pressure affect timings. They exclude tokenization, GPU upload,
context planning and kernel preparation. The original development profile spent
34.25 s converting BF16 values; its disk reads took 1.42 s. The Activity panel now
reports reads and conversion separately. This changes startup work; the final
weight format and inference precision remained F32 in those historical runs.

```bash
mkdir -p models/qwen3-0.6b
for name in config.json tokenizer.json tokenizer_config.json generation_config.json LICENSE model.safetensors; do
  curl --fail --location \
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/$name" \
    --output "models/qwen3-0.6b/$name"
done
cargo build --release
PUPPYGRAD_NVRTC=/path/to/libnvrtc.so.12 \
  target/release/puppygrad llm examples/qwen3_cached.pup \
  --model-dir models/qwen3-0.6b --device cuda:0 \
  --prompt "Explain why the sky is blue in one short sentence." \
  --temperature 0 --max-new-tokens 32 --stream
```

The [CUDA setup](cuda-backend.md) describes the driver and NVRTC requirements.
The [HIP backend](hip-backend.md) uses the same sources and checkpoint with
`--device hip:0`; it includes retained KV state, chunked prefill and automatic
context planning. HIPRTC compilation, deterministic-checkpoint reference parity
and full Qwen3-0.6B text generation are verified on an RX 9070 XT. Controlled
HIP context-throughput measurements use the
[production generation benchmark](../benchmarks/qwen3-hip/generation.py).
Use `--device cpu --cpu-target native --threads 8` for generated C instead.
`emit` only reads checkpoint metadata and requires neither a GPU nor a tokenizer:

```bash
target/release/puppygrad emit examples/qwen3_cached.pup \
  --model-dir models/qwen3-0.6b --sequence-length 24 --backend cuda -o qwen3.cu
```

In the TUI, use `/model qwen3-1.7b` to select the larger model or `/download qwen3-1.7b`
to download its assets. It uses `qwen3_cached.pup` on CPU, CUDA and HIP just like 0.6B.
The official 1.7B checkpoint uses two Safetensors shards and an index; all three
are included in the model manifest. An equivalent command-line run is:

```bash
target/release/puppygrad llm examples/qwen3_cached.pup \
  --model-dir models/qwen3-1.7b --device hip:0 \
  --prompt "Explain what a compiler does." --max-new-tokens 64 --stream
```

The runtime recognizes `model_type: qwen3` and formats its command-line user prompt
using the official tokenizer's non-thinking generation prefix. It displays the
assistant's generated text in both streaming and buffered modes. This is a
single-turn command-line interface; the TUI supports saved multi-turn sessions.
The FFI still accepts ordinary token ID buffers.

The checkpoint loader retains BF16 values in BF16 storage on every backend. When
`tie_word_embeddings` is true, `lm_head.weight` binds to the embedding's existing
input slot, even if both tensors are serialized. Qwen3's
`max_position_embeddings` supplies the provider's context limit. The loader and
metadata-only emitter produce identical bindings for single-file and sharded
checkpoints. With an index, all listed shards are validated before tensor values
are loaded; weights bind in globally sorted name order. Missing files/weights,
incorrect shard assignments, unsafe relative paths and invalid payloads are rejected.
RoPE scaling variants are not implemented by this example.

### Qwen3-1.7B HIP validation

Before native BF16 support, on an RX 9070 XT with F32 weights and KV storage, 512 KV slots, a 33-token
prompt and 64 output tokens, three release runs after one warmup measured:

| Metric | Median |
|---|---:|
| First token | 188.0 ms |
| Decode throughput | 59.5 tokens/s |
| Whole request | 1.247 s |

These warm timings exclude checkpoint loading and initial compilation. They
do not predict speed at larger context capacities. Real HIP checks cover
multi-turn name recall and `FETCH_OLDER` retrieval on both Qwen sizes. Retrieval
feedback keeps the pending question before and after the result so both models
continue answering it after older turns are inserted. Qwen3-1.7B still has small-model
quality limitations; for example, it interpreted the name "Puppy" as an animal persona.

The official [model card](https://huggingface.co/Qwen/Qwen3-0.6B#model-overview)
advertises a **32,768-token context window**, shared by prompt and generated
tokens. The pinned checkpoint config declares `max_position_embeddings: 40960`;
the provider caps its input/output positions at that config value. CUDA
automatically selects a smaller limit when device memory requires it. It is not a validation of model quality beyond the advertised 32K window.
Our original CUDA comparisons covered only 24, 64 and 128 input tokens with
512 KV slots. Use the [context benchmark](../benchmarks/qwen3-cuda/context.py)
to measure larger retained capacities and the workspace cost of prefill.

For this architecture, F32 KV requires `2 * 28 * 8 * 128 * 4` bytes per slot:
224 KiB per token, or 7 GiB at 32K capacity, in addition to approximately
1.11 GiB of native BF16 weights. The production provider rounds required KV capacity up
to a power of two, clipped to the effective FFI context limit (normally at least
512 slots). Attention currently computes over the entire reserved capacity and
masks unused rows. CUDA retained providers now prefill in 128-token chunks by
default. Large single-shot attention tensors can exceed the compiler's 2 GiB
workspace guards even when retained KV and chunked execution fit on the GPU.

Context sizing is runtime policy. The compiler supplies allocation plans; the
provider queries live GPU free memory, reserves 512 MiB for modules, graphs and
transient allocations, and searches for an affordable context within the model's
position limit. `build_model` publishes the resulting limit in the existing
`Info.context_length` field. Each inference checks its actual prefill/tail/decode
buffer requirements against current free memory, accounting for retained buffers
and transient resize peaks. Plans and compiled shapes are cached. `.pup` still
owns state layout and updates; chunk orchestration needs no new language syntax.

Inspect the estimate without loading weight values or running inference:

```bash
target/release/puppygrad llm capacity examples/qwen3_cached.pup \
  --model-dir models/qwen3-0.6b --device cuda:0 --json
```

`--single-shot` estimates whole-prompt prefill; `--prefill-chunk N` selects a
retained chunk size. `--memory-budget-mib N` enables offline planning without a
GPU, and `--reserve-mib N` changes reserved headroom. This command uses the same
planner as automatic provider creation. Its result is an allocation estimate,
not a model-quality test or a guarantee against concurrent GPU allocation.
Arbitrary source branches can make memory use non-monotonic; actual request
shapes are checked separately.

At the FFI construction boundary, optional JSON fields `auto_context` (defaults
to true on CUDA and false on CPU), `prefill_chunk`, `context_reserve_mib` (512),
and `context_budget_mib` control this provider policy. A configured budget can
only reduce live available memory. CPU retains its existing context policy;
explicit CPU chunking is supported for retained programs. Shared-library
providers continue to publish their own context limits through the same ABI.

`--verify-reference` remains a GPT-2-only check. Qwen is verified by a deterministic
small BF16 checkpoint against an independent scalar reference, and by full-model
CUDA logits and greedy token parity against native tinygrad.

The cached example owns keys, values, and position through generic `state`, `store`,
and `after` operations. It processes the prompt once and then a single new token
per execution. State stays on the GPU and resets for a new `infer`; the LLM harness
knows no cache layout. [Retained-state semantics](retained-state.md) describe the
contract and lifetime. `examples/qwen3.pup` remains the full-prefix baseline.

## Original full-prefix CUDA comparison

The reusable [benchmark harness](../benchmarks/qwen3-cuda/README.md) uses tinygrad's
actual [Transformer implementation](https://github.com/tinygrad/tinygrad/blob/1a58c3ae9d5ff5605d81085cf1a364c113e95cf6/tinygrad/llm/model.py), at revision
`1a58c3ae9d5ff5605d81085cf1a364c113e95cf6`. Its current CLI entrypoint is
`python -m tinygrad.llm` and [lists `qwen3:0.6b`](https://github.com/tinygrad/tinygrad/blob/1a58c3ae9d5ff5605d81085cf1a364c113e95cf6/tinygrad/llm/cli.py); `examples/llm.py` no longer exists
in that revision.

Those historical comparison runs used identical BF16 checkpoint values expanded to F32, tied
embedding/output storage, and F32 attention. Tinygrad normally uses F16 KV storage;
the harness explicitly selects F32 KV for matching arithmetic. This compares
compiler/runtime execution, rather than Puppygrad F32 against the CLI's usual
quantized GGUF weights. Tinygrad uses `BEAM=0` and `TinyJit`; Puppygrad uses the
production release CUDA executor.

Full-prefix runs start at position zero and process all input tokens on every
call. The generation comparison additionally runs tinygrad's native cached decode.
Its configured context capacity is 512, with 32-token prefill chunks. Generation
uses the actual Puppygrad FFI, including token binding, host sampling and callbacks,
and native tinygrad `generate`, including GPU sampling and token fetch.
The original `examples/qwen3.pup` baseline retains weights, scratch and compiled
shapes, but recomputes the full prefix for each generated token. Warm timings exclude checkpoint loading,
compilation and warmup; new prefix lengths still need compilation on a cold run.

Raw samples, numerical checks, retained-memory counters and compilation times are
written to `.cache/qwen3-compare/results.json`.

Measured on an RTX 2070 (F32, three warmups and eleven samples):

| Prefix tokens | Puppygrad | tinygrad |
|---:|---:|---:|
| 24 | 49.00 ms | 45.77 ms |
| 64 | 88.11 ms | 91.28 ms |
| 128 | 163.60 ms | 169.24 ms |

All full-logit checks stayed below 0.0001 maximum absolute error, and eight greedy
token IDs matched exactly. Actual FFI generation took 396.03 ms for eight tokens
versus 622.61 ms for native tinygrad on the 24-token chat prompt. Tinygrad's
cached decode alone was faster: 25.65 versus 49.52 ms per token. Its
first-token path cost more here; this short-run total does not predict longer
generation speed. The complete report is `.cache/qwen3-compare/comparison.md`.


## Retained generation results

The retained source and generic CUDA warp contractions achieve **2.30x faster
cached decode** than native tinygrad on the RTX 2070, using equal F32 weights and
F32 KV capacity 512 (three warmups, eleven paired samples):

| Metric | Puppygrad cached | tinygrad cached |
|---|---:|---:|
| First token, fresh infer | 53.64 ms | 439.69 ms |
| Decode token | 11.14 ms | 25.59 ms |
| Eight tokens total | 131.66 ms | 618.87 ms |

Eight greedy IDs match exactly; full logits remain below 0.0001 maximum absolute
error. State uses 112 MiB, warm decode allocates no device buffers and uploads four
bytes of token input. The cache algorithm is in `.pup`; generic compiler changes
provide retained writable buffers, ordered STORE/AFTER operations and warp-parallel
small contractions. Fresh fixed-prefix forwards are still slower here because
the source attends over reserved capacity. Native tinygrad's symbolic/chunked
first-token path costs more than its fixed-prefix wrapper, so short-request total
speedup should not be extrapolated to other generation lengths or configurations.
Cold NVRTC compilation still takes roughly 22–23 seconds per new prompt shape.

Run `python benchmarks/qwen3-cuda/benchmark.py --source examples/qwen3_cached.pup
--output-dir .cache/qwen3-cached` after building release. Raw results are
`.cache/qwen3-cached/results.json`; the detailed report is
`.cache/qwen3-cached/comparison.md`.
