# Qwen3-0.6B in Puppygrad

[examples/qwen3_cached.pup](../examples/qwen3_cached.pup) implements the dense Qwen3 decoder
using ordinary source functions and primitive tensor Pops. It runs through the
existing `llm` application and LLM buffer/FFI contract, on generated C CPU kernels
or generated CUDA kernels. There is no Qwen model computation in Rust or Python.

The source includes RMSNorm, per-head query/key normalization, half-split RoPE,
grouped causal attention, SiLU gated feed-forward blocks and tied output weights.
The bundled [source library](../stdlib/nn.pup) provides `rms_norm`,
`rotary_positions`, and `grouped_query_attention`; these expand into existing
operations and introduce no language syntax or model-specific compiler lowering.
Qwen3-0.6B has 16 query heads and 8 KV heads, each 128 wide. Its 1024-wide hidden
state does **not** imply a 64-wide attention head.

## Assets and execution

The tested checkpoint is the official
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its Apache-2.0 license is retained
alongside the downloaded model. Assets, generated code, PTX and benchmark logs
remain under ignored `models/` and `.cache/` directories.

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
Use `--device cpu --cpu-target native --threads 8` for generated C instead.
`emit` only reads checkpoint metadata and requires neither a GPU nor a tokenizer:

```bash
target/release/puppygrad emit examples/qwen3_cached.pup \
  --model-dir models/qwen3-0.6b --sequence-length 24 --backend cuda -o qwen3.cu
```

The runtime recognizes `model_type: qwen3` and formats its single user prompt
using the official tokenizer's non-thinking generation prefix. It displays the
assistant's generated text in both streaming and buffered modes. This is a
single-turn text interface; the FFI still accepts ordinary token ID buffers.

The checkpoint loader expands BF16 values to F32 for the current backends. When
`tie_word_embeddings` is true, `lm_head.weight` binds to the embedding's existing
input slot, even if both tensors are serialized. Qwen3's
`max_position_embeddings` supplies the provider's context limit. The loader and
metadata-only emitter produce identical bindings. Sharded safetensors and RoPE
scaling variants are not implemented by this example.

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
2.22 GiB of F32 weights. The production provider rounds required KV capacity up
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

Both implementations use identical BF16 checkpoint values expanded to F32, tied
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
