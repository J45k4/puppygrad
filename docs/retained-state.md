# Retained buffers in .pup

`state(name, dtype, shape)` declares a zero-initialized writable tensor that survives
executions. `store(index(buffer, rows), value)` updates selected rows; `after(value,
write, ...)` makes those writes explicit dependencies of a later read or output.
These are generic buffer operations, usable for counters, recurrent networks,
rolling signals, optimizer state, or transformer caches.

```python
counter = state("counter", i32, [1])
previous = load(index(counter, cast(0, i32)))
write = store(index(counter, cast(0, i32)), previous + 1)
output load(index(after(counter, write), cast(0, i32)))
```

The compiler represents writes as distinct STORE Pops and ordering as AFTER Pops.
Values remain a DAG, and unused writes are rejected: attach each write to an
output through `after`. Store destinations must index a state allocation or its
contiguous reshape/ordered view. Payload shape and dtype must match the indexed
rows. Bounds are checked. Repeated row indices are applied in source order, so the
last update wins; overlapping copies snapshot their source before writing.
Parallel CUDA stores require proven distinct indices; otherwise they execute
serially. Differentiation through effects is not supported.

CPU and CUDA runtimes allocate and retain state separately from read-only inputs,
outputs, and temporary arenas. CUDA writes stay on the device; only declared
outputs are downloaded. `cpu::Runtime` and `cuda::Runtime` share allocations across
compiled input-shape specializations. `reset_state()` zeros retained buffers; drop
all runtime/executable owners to free them. Changing an allocation's dtype or size
replaces it with zeroed storage. Programs sharing a runtime must use consistent
state declarations and slot layouts. Separate model instances have separate
runtimes. A failed execution can have applied earlier writes: reset before reuse
when an error invalidates the state.

## LLM use

[examples/qwen3_cached.pup](../examples/qwen3_cached.pup) implements the KV cache,
rotary offsets, causal attention, and position updates entirely in source. The
backend sees ordinary state tensors and writes; it knows no attention heads,
Qwen layers, or cache policy. The generic LLM adapter resets state at the beginning
of each `infer`. A source with state declarations opts into a retained input
stream: the first execution receives the complete prompt; subsequent executions
receive only the newly sampled token. A source without state declarations keeps
the full-prefix convention.

`config("buffer_capacity")` supplies a generic retained-buffer capacity, at least
512 where the checkpoint context limit permits, rounded up to fit the requested
prompt and generation. Metadata-only emission defaults to 512 or the smaller
checkpoint limit. The source controls what this capacity means and the shapes of
its allocations. The LLM application still uses `build_model`, `infer`,
`read_output`, and `free_model` with an opaque model handle; it has no KV layout or
prefill/decode model entrypoints.

The source specializes input length and capacity, so a typical generation needs
one prompt graph and one one-token graph. Weights and state are shared between
these graphs. Cold graph compilation remains separate from execution; a new
prompt length or capacity can require compilation. The example attends over its
reserved capacity and masks unused rows. Cache eviction, sliding windows, and
cross-infer prefix reuse are source/application policies not implemented here.
