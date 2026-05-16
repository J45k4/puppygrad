# Shared Model Runtime

Puppygrad keeps reusable autoregressive runtime pieces under `src/models/` and leaves concrete transformer blocks under each model directory until more than one native model exists.

Current shared pieces:

- `models::generation`: generation CLI args, sampling config, sampling helpers, generic generation stats, and profile attachment.
- `models::streaming`: token decoder trait and incremental text streamer.
- `models::autoregressive`: minimal autoregressive decoder and KV-cache traits.
- `models::assets`: Hugging Face asset path, required-file, and download orchestration helpers.
- `models::config`: typed JSON config loading.
- `models::safetensors`: tensor store and tensor validation helpers.
- `models::cpu`: reusable CPU math kernels.

## Autoregressive Generation

Models that can prefill a prompt, decode one token at a time, and pick a next token can implement `AutoregressiveDecoder`:

```rust
use puppygrad::models::autoregressive::{generate, AutoregressiveDecoder, KvCache};

struct MyCache {
    seq_len: usize,
    max_seq_len: usize,
}

impl KvCache for MyCache {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
    }
}

struct MyModel;

impl AutoregressiveDecoder for MyModel {
    type Cache = MyCache;
    type Logits = Vec<f32>;
    type Error = std::convert::Infallible;

    fn max_context_len(&self) -> usize {
        1024
    }

    fn new_cache(&self) -> Result<Self::Cache, Self::Error> {
        Ok(MyCache {
            seq_len: 0,
            max_seq_len: 1024,
        })
    }

    fn prefill(
        &self,
        _input_ids: &[usize],
        _cache: &mut Self::Cache,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(vec![0.0, 1.0])
    }

    fn forward_one(
        &self,
        _token_id: usize,
        _cache: &mut Self::Cache,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(vec![1.0, 0.0])
    }

    fn select_next_token(&self, logits: &Self::Logits) -> Result<usize, Self::Error> {
        Ok(puppygrad::models::generation::argmax_logits(logits))
    }
}

let model = MyModel;
let output_ids = generate(&model, &[0], 3, |_token_id| Ok::<(), std::convert::Infallible>(()))?;
assert!(!output_ids.is_empty());
```

Text streaming stays separate from model execution. Implement `TokenDecoder` for a tokenizer, then feed generated token ids through `IncrementalTextStreamer` to emit text chunks with a full-decode fallback for tokenizer edge cases.
