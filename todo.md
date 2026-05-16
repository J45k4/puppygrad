# Puppygrad TODO

## GPT-2 Rust Backend Performance

Already implemented:

- Persistent worker thread pool.
- Parallel dense projections for attention QKV, attention projection, MLP `c_fc`, MLP `c_proj`, and final vocab projection.
- Transposed dense weights at model construction for contiguous projection dot products.

Remaining optimization ideas:

- Avoid copying input activations into `Arc<[f32]>` for every parallel dense call.
  - Current parallel helpers clone small activation buffers so jobs can be `'static`.
  - Options: add a scoped parallel API, use owned scratch buffers with stable lifetimes, or specialize model execution around borrowed worker scopes.

- Reuse scratch buffers during generation.
  - Reuse storage for layernorm outputs, QKV, attention output, MLP activations, projection outputs, softmax scores, and logits.
  - This should reduce allocator pressure and make token timings more stable.

- Parallelize attention heads.
  - Current dense projections are parallel, but attention score/value accumulation is still serial over heads.
  - GPT-2 small has 12 heads, which should map reasonably well to worker tasks, especially for longer contexts.

- Improve cached attention memory layout.
  - Review KV cache layout for cache-friendly reads during attention.
  - Consider per-head contiguous key/value blocks if it improves sequential access.

- Decode streamed text more efficiently.
  - Current streaming path decodes the accumulated token list each step to compute deltas.
  - Prefer decoding only new tokens where safe, with a fallback for tokenizer edge cases.

- Add SIMD dot-product kernels.
  - After transposed weights, dense projection rows are contiguous and easier to vectorize.
  - Add portable scalar fallback plus platform-specific paths such as NEON on Apple Silicon and AVX/FMA on x86.

- Tune dense chunk sizes per operation.
  - `linear_block` and logits projection use fixed chunk sizes today.
  - Tune chunk sizes separately for QKV, MLP expand, MLP project, attention project, and vocab projection.

- Make parallel thresholds configurable or adaptive.
  - Current dense parallelism uses a fixed work-size threshold.
  - Measure thresholds for different thread counts and model sizes.

- Add benchmark statistics beyond averages.
  - Include min, median, p95, max, and standard deviation.
  - Average-only results hide scheduling noise.

- Add prompt-file experiment support.
  - Allow benchmarking multiple prompts and context lengths from a text file.
  - Report per-prompt and aggregate performance.

- Add operation-level profiling.
  - Track time spent in layernorm, QKV projection, attention, MLP, final logits, tokenization, and decoding.
  - This will show which optimization to do next with less guessing.

- Consider quantized weights later.
  - Start with simple int8 or fp16 storage experiments for memory bandwidth.
  - Keep the current f32 path as the correctness/reference backend.
