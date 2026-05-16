# Puppygrad TODO

## GPT-2 Rust Backend Performance

- [x] Persistent worker thread pool.
- [x] Parallel dense projections for attention QKV, attention projection, MLP `c_fc`, MLP `c_proj`, and final vocab projection.
- [x] Transposed dense weights at model construction for contiguous projection dot products.
- [x] Avoid copying input activations into `Arc<[f32]>` for every parallel dense call by adding scoped worker jobs over borrowed activation buffers.
- [x] Reuse scratch buffers during cached generation for layernorm outputs, QKV, attention output, MLP activations, projection outputs, and logits.
- [x] Parallelize attention heads for full and cached attention when attention work is large enough.
- [x] Improve cached attention memory layout with per-head contiguous key/value blocks.
- [x] Decode streamed text more efficiently by decoding only the new token on the common safe path, with accumulated-token fallback for tokenizer edge cases.
- [x] Add SIMD dot-product kernels with a portable scalar fallback, NEON on Apple Silicon/aarch64, and AVX on x86/x86_64.
- [x] Tune dense chunk sizes per operation by making QKV, MLP expand, MLP project, attention project, and vocab projection chunk sizes separately configurable.
- [x] Make parallel thresholds configurable and sweepable from `experiment gpt2`.
- [x] Add benchmark statistics beyond averages: min, median, p95, max, and standard deviation.
- [x] Add prompt-file experiment support with per-prompt and aggregate performance rows.
- [x] Add operation-level profiling for layernorm, QKV projection, attention, MLP projections, final logits, tokenization, and decoding.
- [x] Add an experimental row-wise int8 weight path while keeping the current f32 path as the correctness/reference backend.
- [x] Draw a chart of benchmark improvements over time from `benchmarks/gpt2_experiment_history.csv`.
