# Puppygrad TODO

## Shared Model Runtime Extractions

- [x] Extract shared CLI generation arguments with `clap::Args`, including `max_new_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `repeat_penalty`, and `repeat_last_n`.
- [x] Add a generic token streaming module with a tokenizer/decoder trait and an incremental text streamer reusable by GPT-2, Qwen, Llama, and similar text models.
- [x] Extract generic generation stats for prompt tokens, generated tokens, tokenization time, prefill time, decode time, time to first token, and token/sec helpers.
- [x] Keep model-specific operation profiles separate for now, but make them attachable to generic generation stats.
- [ ] Expand shared Hugging Face asset utilities for required-file checks, model directory resolution, cache path conventions, and download orchestration.
- [ ] Add generic JSON config loading helpers for model config files.
- [ ] Improve safetensors helpers with a tensor store API for required tensors, optional tensors, dtype checks, and shape validation.
- [ ] Extract reusable CPU math kernels from GPT-2, including dot product, dense projection, transposed dense projection, row-wise quantized matvec, layernorm, GELU, softmax, and causal attention helpers where appropriate.
- [ ] Introduce a minimal KV cache trait for shared cache concepts like `seq_len`, `max_seq_len`, and `clear`, without forcing a common memory layout yet.
- [ ] Delay broader transformer block extraction until there is at least one second real model, so GPT-2 learned positions and Qwen/Llama RoPE do not get forced into the wrong abstraction.
- [ ] Review whether `Gpt2RustConfig` can be split into generic CPU backend options and model/op-specific tuning options.
- [ ] Move generic autoregressive generation examples and documentation out of GPT-2-specific docs.
