# LLM runtime and FFI contract

`src/runtime/llm.rs` is the LLM application: it reads command arguments, tokenizes
prompts, and displays generated text. It talks to the model through the C ABI in
[puppygrad_llm.h](../include/puppygrad_llm.h), implemented by
[src/runtime/llm_ffi.rs](../src/runtime/llm_ffi.rs).

The compiler does not define the LLM application contract. A provider owns model
construction, generation, sampling, output storage, and cleanup. The runtime passes
settings and chooses whether to register a token callback.

## Command line

`llm MODEL`, `llm run MODEL`, and `run MODEL` execute the same runtime.
`llm benchmark [MODEL]` measures CPU thread scaling:

```sh
cargo run --release -- llm examples/llm.pup \
  --model-dir models/gpt2 --device cpu \
  --prompt "The meaning of life is" --max-new-tokens 8 \
  --stream --temperature 0.8 --seed 42

cargo run --release -- llm ./model.so \
  --model-dir ./model-assets --prompt "Hello" --max-new-tokens 32
```

- `--stream` registers `on_tokens` and flushes decoded text incrementally. Without
  it, no token callback is registered; the runtime queries output after completion.
- `--temperature 0` selects greedy generation (the default). Positive values sample
  from the temperature-scaled distribution. Temperature must be finite and nonnegative.
- `--seed` supplies the provider's sampling seed. Streaming must not affect sampling.
- `--max-new-tokens` limits newly generated tokens; zero returns the prompt unchanged.
- `--model-dir` supplies a Hugging Face `tokenizer.json` to this application. The
  provider decides what other model assets it needs in that directory.
- `--threads N` requests CPU GEMM parallelism. The `.pup` provider defaults to
  `min(available CPUs, 8)`; `1` uses serial GEMM. Counts above 8 are allowed. Other
  generated C loops remain serial. The generated library owns a POSIX thread pool
  per invocation, reusing it across contractions. The caller participates, so N
  means at most N executing threads, capped by available output tiles. No Rust
  GEMM callback or Rayon pool participates in generated computation.
- `--device` is passed to the provider. The `.pup` provider supports CPU/C and
  `cuda:<index>` (or `cuda` for index 0). The initial [CUDA backend](cuda-backend.md)
  requires an NVIDIA driver and NVRTC at execution time. `--threads` controls CPU
  execution only; nondefault `--cpu-target` settings are rejected for CUDA.
- `--cpu-target generic|native|avx2` selects the C compiler's instruction target
  for `.pup` graphs. The default is `generic`; `native` enables this host's CPU
  features and `avx2` enables AVX2 on supporting x86 hosts. The same option is
  forwarded to benchmark workers, whose JSON results record compiler settings.
  A precompiled shared library already has its instruction target; nondefault
  target selection is rejected for that input.
- `--verify-reference` checks the `.pup` GPT-2 adapter against the native GPT-2 model;
  it is not part of the generic LLM contract. It compares logits and their greedy
  choices even when actual generation uses a positive temperature.

Text goes to stdout; diagnostics go to stderr. The application uses an incremental
Hugging Face tokenizer decoder, and flushes any final undecoded byte sequence at
completion. Tokenizer IDs must fit the model vocabulary; padded model vocabularies
are allowed, but generated IDs must exist in the supplied tokenizer.

## Provider API

A shared library exports `get_llm_api()`, returning a pointer to a stable `PupLlmApi`
function table. The runtime checks `abi_version`, `struct_size`, and required function
pointers before constructing the model. API v1 consists of:

| Function | Contract |
| --- | --- |
| `build_model` | Borrow UTF-8 JSON configuration; return opaque owned state and fill vocabulary size, EOS ID, and context limit. Return NULL on failure. |
| `infer` | Borrow prompt IDs, generation settings, and callbacks. Generate tokens, retain output, and report terminal status. Return 0 on success, nonzero on failure. |
| `read_output` | Query/copy the newly generated IDs, excluding the prompt. A NULL destination with capacity 0 queries the required count. |
| `free_model` | Release state and every resource it owns. Called exactly once for a successful build. No callbacks or work may survive its return. |

The JSON configuration currently contains `model_dir`, `device`, `threads`,
`source`, `verify_reference`, and `cpu_target` (default `generic`). These are JSON
values passed as bytes across the ABI. External providers may ignore adapter-specific fields. Configuration
is borrowed only during construction; providers must copy anything they retain.

`PupLlmGeneration` carries a u64 token limit, f32 temperature, u64 seed, and a reserved
u32 that must be zero. EOS ID `UINT32_MAX` means no EOS token. The context limit
bounds the number of input tokens an inference step can consume: an N-token prompt
and M generated tokens require at most `N + max(M - 1, 0)` positions. Providers
must reject invalid settings and IDs even when called outside this Rust wrapper.

The same state can be reused for multiple `infer` calls. In v1 each call starts an
independent generation from the complete supplied prompt, replacing retained output.
Weights and compiled artifacts may remain resident between calls. Incremental
conversation-state semantics are not implicit in this ABI.

## Callbacks, errors, and ownership

**ABI v1 is synchronous:** `infer` returns after computation and callbacks finish.
Callbacks execute on the calling thread. A caller can run it on its own worker
thread, but must not concurrently call the same state or reenter it from callbacks.
There are no hidden background workers left running when `infer` returns.

- `on_tokens` is optional. It receives an ordered batch of IDs whose pointer is
  borrowed only for that callback. A provider may batch tokens; it must retain the
  same sequence for `read_output`. The callback has no ownership of model memory.
- `on_done` is optional for native callers; the Rust wrapper always registers it.
  With valid ABI pointers it fires exactly once, after all token callbacks, including
  on validation/execution failure. Reasons are limit reached, EOS, or error.
- Completion with an error also requires a nonzero `infer` return status. Successful
  completion requires output ending in EOS or reaching the requested token limit.
- `user` is an opaque caller pointer, passed through unchanged. It and all borrowed
  arguments must remain valid until `infer` returns. Callbacks must not unwind across
  the C ABI. Rust callback panics are caught and reported by the wrapper.
- `read_output` always reports the required count. For a non-query call, insufficient
  capacity returns an error and does not write beyond the buffer. Output remains
  stable until the next `infer` or `free_model` call.
- Errors use caller-owned `PupLlmError` storage. Write at most `capacity` UTF-8 bytes
  and set `length` to bytes written; no terminator is required. Set length to zero
  when no message is supplied. Truncation is allowed.
- If construction fails, the provider releases any partially constructed resources.
  Successful state must be released through `free_model`, never the host allocator.

The Rust wrapper checks callback ordering, completion, token IDs, token limits, and
streamed/retained output agreement. It keeps the shared library loaded until
`free_model` returns. A bad native pointer or memory overwrite cannot be repaired by
these protocol checks; providers must honor the C memory contract.

Asynchronous completion, cancellation, reset, and per-request handles are not part
of v1. They require an explicit future contract rather than changing these lifetimes
silently. All ABI sizes use the host architecture; libraries must match the host ABI.

## Current .pup integration

`src/models/pup_llm.rs` adapts current `.pup` graphs to this same function table. It
loads a GPT-2 or Qwen3 safetensors checkpoint, compiles the model's graph to C or CUDA C++, and
retains weights and compiled shape specializations until `free_model`. CUDA shape
variants share one runtime: weights remain resident, scratch capacity is reused,
and each subsequent forward uploads only fresh token IDs. `free_model` releases
the runtime's device allocations as well as every compiled module. The token-generation loop
and sampling run in the adapter. Model math remains in
[examples/llm.pup](../examples/llm.pup),
[examples/qwen3.pup](../examples/qwen3.pup), and the source standard library.
The [Qwen3 guide](qwen3.md) describes checkpoint binding, non-thinking prompts,
and the CUDA comparison against tinygrad.

This adapter is statically linked into the executable and called through C ABI
function pointers. External model libraries use `get_llm_api` via dynamic loading.
The `.pup` compiler does **not yet generate this complete LLM shared-library ABI**:
arbitrary `@export`, model lifecycle exports, and generation-loop lowering remain
future compiler work. Generic [retained state and STORE/AFTER operations](retained-state.md)
are implemented. `examples/llm.pup` and `examples/qwen3.pup` recompute the full prefix;
`examples/qwen3_cached.pup` owns its KV algorithm in source. The adapter resets generic
state per `infer` and sends only new tokens after the first retained execution.
The application and external ABI have no KV layout or prefill/decode entrypoints.

## Verification

`cargo test --test llm_ffi` builds an independent C provider against the public header
and loads it dynamically. Tests exercise ABI mismatches, lifecycle cleanup, streaming
and buffered output, sampling settings, EOS, zero-token requests, provider failures,
and callback protocol violations. A tiny checkpoint test runs the `.pup` adapter
through the same ABI and checks deterministic sampling and reuse of retained state.


## CPU thread benchmark

```sh
# Every thread count from 1 to the available CPU maximum:
cargo run --release -- llm benchmark

# Explicit program and a smaller sweep:
cargo run --release -- llm benchmark examples/llm.pup \
  --model-dir models/gpt2 --max-threads 6 \
  --runs 3 --warmups 1 --max-new-tokens 8 \
  --prompt "The meaning of life is"
```

The default program is `examples/llm.pup`; a shared library implementing the LLM
contract also works. The benchmark currently supports `--device cpu`. Providers
receive `threads` in their build configuration; how they implement CPU parallelism
is provider-specific.

The sweep tests every count from 1 through `--max-threads` (available CPU parallelism
by default), in a reproducibly shuffled order. Each count uses a fresh process
with `threads` passed to the provider and tokenizer parallelism disabled. Weights load
once per count. Untimed generations warm every shape specialization before timed
runs. Sampling is greedy with seed 42 and no streaming or reference verifier.
Timing covers the complete `infer` call, including sampling and output copying;
it excludes model/tokenizer loading, text decoding, and compilation/warmup.
All generations must produce identical token IDs. Early EOS is an error rather
than silently measuring a shorter workload; choose another prompt or fewer tokens.

A sorted table reports median generation time, min–max, tokens/second, and speedup
relative to one thread. Raw samples, per-worker logs, host/load metadata, CSV, JSON,
and Markdown are saved under ignored `.cache/benchmarks/`. `--output-dir PATH`
selects a new directory; existing directories are rejected to preserve old results.
The old standalone Rust example and Python sweep are now part of this command;
no Python helper or separately built example is needed. Use a release build for
meaningful performance measurements. A debug build prints a reminder and is marked
in the metadata. Host load can affect the ranking; inspect ranges and metadata.
