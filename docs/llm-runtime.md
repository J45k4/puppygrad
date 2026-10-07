# LLM runtime and FFI contract

`src/runtime/llm.rs` is the LLM application: it reads command arguments, tokenizes
prompts, and displays generated text. It talks to the model through the C ABI in
[puppygrad_llm.h](../include/puppygrad_llm.h), implemented by
[src/runtime/llm_ffi.rs](../src/runtime/llm_ffi.rs).

The compiler does not define the LLM application contract. A provider owns model
construction, generation, sampling, output storage, and cleanup. The runtime passes
settings and chooses whether to register a token callback.

## Command line

Running `puppygrad` with no arguments in a terminal opens the interactive UI
(`cargo run --release` from a checkout). Explicit `puppygrad tui` accepts
`--device`, `--cache-dir`, and `--catalog`. Without a terminal, the no-argument
command prints help; existing command-line operations remain available for scripts.

Type `/model` to browse GPT-2 small and Qwen3-0.6B. The list distinguishes downloaded,
partial, and missing checkpoints. Use arrow keys to choose an entry; Enter downloads
missing files or selects a downloaded model. `d` downloads the highlighted model.
Downloads run in the background with file/byte progress, and completed files are
published atomically. `/download [id]` also downloads assets directly.

Enter a prompt to stream its response. The selected model stays loaded between
requests, while each prompt is currently independent (conversation history is not
added automatically). Changing the model or device frees the previous provider
before loading the next one. `/device cpu|cuda:0|hip:0`, `/temperature N`, and
`/tokens auto|N` adjust execution; `/help` lists commands. Esc stops an operation between
download progress updates or generation callbacks, and `/quit` or Ctrl-C without
a composer selection exits
after the current operation returns. The terminal is restored on normal exit and
errors. First-response time and overall tokens/sec include model loading and cold
compilation where applicable.

Enter sends the prompt; Shift+Enter inserts a newline. The composer grows up to
six lines and scrolls to keep the cursor visible. Pasting preserves line breaks.
Modified keys use the terminal's enhanced keyboard protocol; Ctrl+J also inserts
a newline on terminals that cannot distinguish Shift+Enter.
Arrow keys move the composer cursor, including between wrapped lines. Home/End
move to the current displayed line's edges; Ctrl+Home/End move to the whole
prompt's start/end. Typing, pasting, Backspace and Delete edit at the cursor.
Shift+arrows extend a selection, and Ctrl+A selects the entire prompt. Ctrl+Left
stops at word boundaries, including the previous word's end when starting at a
word's start; Ctrl+Right moves to a word's end. Adding Shift selects to that point.
Backspace/Delete removes selected text, and typing or pasting replaces it.
Ctrl+C copies selected text, Ctrl+X cuts it, and Ctrl+V pastes from the desktop
clipboard. Esc cancels a selection when idle. Clipboard shortcuts use `wl-copy`
and `wl-paste` on Wayland, `xclip` on X11, or `pbcopy`/`pbpaste` on macOS. Terminal
paste with Ctrl+Shift+V also works, including without these clipboard tools.

The loaded model's tokenizer is also reused between prompts.
When the TUI opens, it warms the selected downloaded model in the background:
tokenizer and weight loading, kernel compilation, and small decode/prefill runs.
Warmup output is discarded. You can type immediately and submit a prompt to wait
behind warmup. Selecting another model or device starts its warmup as well; no
assets are downloaded automatically. Esc stops warmup. `--no-warmup` keeps model
loading deferred until the first prompt. Warmup uses the same live-memory checks
and `--max-memory` limit as normal inference. Optional prefill warmup is best-effort;
a failed larger dummy shape does not discard a working decode provider.
Retained TUI models automatically reuse 1-, 8-, and 128-token prefill programs.
A 17-token prompt runs as 8 + 8 + 1; tails consume only real tokens, with no
padding or extra position advances. Short requests reuse the warmed 8-token and
decode programs. Longer requests can compile the 128-token program once per
context capacity; cache growth may also require new programs. Allocation plans
are cached per execution shape, so another prompt length using the same shapes
does not rebuild the graph. Explicit provider prefill chunks keep their previous
behavior; models without retained state still use complete-prefix execution.

Responses have no default output-token cap in the TUI. They continue until EOS,
Esc, or the model's context or device-memory limit. `/tokens N` sets an explicit
response cap; `/tokens auto` removes it. The footer distinguishes token, context,
and memory exhaustion from normal completion. Partial text remains visible when
available memory prevents further growth.

The UI initially plans a 512-token bucket for the prompt, reducing optional
headroom and automatic prefill chunks if necessary. Generation grows buffers on
demand without reloading weights or recomputing earlier tokens. Retained tensor
coordinates and contents survive growth; model source still owns the cache layout
and position updates. GPU growth currently stages one retained tensor through host
memory at a time, so crossing a capacity boundary can briefly pause streaming.
The next request resets retained contents but reuses the grown capacity.

Providers check live GPU memory before requests and allocation growth. The
`--max-memory` buffer limit includes replacement overlap, and initial planning
reserves space for modules and other runtime overhead. The complete prompt must
fit; optional output stops gracefully when the next context allocation cannot fit.
Reducing context cannot make oversized model weights fit.
It does not run the maximum-context search before a short prompt;
`llm capacity` remains the explicit way to probe that limit. Request allocation
planning retains the lowered kernels for compilation instead of lowering twice.
Checkpoint loading reads and converts one tensor at a time to avoid keeping a
second full checkpoint in host memory.

Set `PUPPYGRAD_STARTUP_PROFILE=1` to record checkpoint loading, context planning,
request planning and per-shape compilation times in `tui.log`. Release builds are
still recommended; ordinary development builds now optimize Puppygrad's graph
and checkpoint code as well.

`puppygrad --max-memory 10` opens the UI with a 10 GiB GPU model-buffer budget.
The global flag also works with `tui`, `run`, `llm run` and `llm capacity`, before
or after the command. Bare numbers are GiB; explicit sizes such as `2.5GiB`,
`512MiB` and `2GB` are accepted. The cap covers this provider's weights, retained
state, scratch and output buffers, including temporary overlap during resizing.
Physical free memory remains an additional constraint. Driver modules, other
GPU applications and host RAM are outside the cap; the capacity planner's
driver reserve is separate. CPU providers and external shared libraries currently
reject this flag rather than silently ignoring it.

Checkpoint metadata lives beside its program in
[llm.model.json](../examples/llm.model.json) and
[qwen3-0.6b.model.json](../examples/qwen3-0.6b.model.json), with repository, pinned
revision, and required filenames. The runtime builds Hugging Face URLs; `.pup`
contains only the computation. Built-in manifests and programs are embedded in the
binary, so the UI also works outside the checkout. `--catalog DIR` reads custom
`*.model.json` files and their relative `.pup` programs (for example `--catalog examples`).

Existing `models/<id>` directories are reused when present. Otherwise checkpoints
live under `$XDG_CACHE_HOME/puppygrad/models/<id>/<revision>`, falling back to
`~/.cache/puppygrad`. Programs, compiled kernels, and `tui.log` diagnostics use the
same UI cache root; `--cache-dir` overrides it. An existing local directory is
treated as caller-provided assets; its revision is not independently verified.

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
  `cuda:<index>` (or `cuda` for index 0) and `hip:<index>` (or `hip`). The initial [CUDA backend](cuda-backend.md)
  requires an NVIDIA driver and NVRTC at execution time. `--threads` controls CPU
  execution only; nondefault `--cpu-target` settings are rejected for GPU execution.
  The [HIP backend](hip-backend.md) uses HIPRTC and the AMD HIP runtime with the same model graph and retained-state contract.
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
  on validation/execution failure. Reasons are token limit (0), EOS (1), error (2),
  model context exhausted (3), or device-memory budget exhausted (4).
- Completion with an error also requires a nonzero `infer` return status. Successful
  completion requires EOS, the requested token limit, the model context limit, or
  a memory stop retaining the partial output. ABI structure layouts are unchanged.
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

The built-in .pup provider cooperatively stops after a Rust host token callback
fails (including TUI cancellation). Foreign providers still own their execution
loop; the native void callback has no cancellation return value.

Asynchronous completion, cancellation, reset, and per-request handles are not part
of v1. They require an explicit future contract rather than changing these lifetimes
silently. All ABI sizes use the host architecture; libraries must match the host ABI.

## Current .pup integration

`src/models/pup_llm.rs` adapts current `.pup` graphs to this same function table. It
loads a GPT-2 or Qwen3 safetensors checkpoint, compiles the model's graph to C, CUDA C++ or HIP C++, and
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
