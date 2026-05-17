# Puppygrad

Rust-first deep learning sandbox with:
- an in-house tinygrad-inspired computation engine, and
- native model runtime experiments built directly in Rust.

## Build

```bash
cargo build --release
```

## Models

Implemented:

| Model | Status | Runtime | Notes |
| --- | --- | --- | --- |
| GPT-2 | Working | Rust reference | Loads Hugging Face `config.json`, `tokenizer.json`, and `model.safetensors`; uses greedy/sampled decoding, token streaming, and a KV cache. |
| Whisper | Working MVP | Rust reference | Prepares Hugging Face Whisper assets, loads typed model/preprocessor/tokenizer/weight metadata, decodes PCM WAV input, computes log-mel features, and runs greedy `tiny.en`/`tiny` transcription. |
| Qwen | Stub | None yet | CLI placeholder for future native loading/runtime work. |

Model assets are stored under the project-root `models/` directory, which is ignored by git. Rust source lives under `src/models/` and is tracked. GPT-2-specific code is organized under `src/models/gpt2/`, with the current Rust reference implementation in `src/models/gpt2/rust.rs`.

Shared model runtime code is intentionally limited to pieces that already have clear cross-model shape: generation CLI args and sampling config, token streaming, generation stats, asset/config loading, safetensors access, CPU math kernels, and minimal autoregressive/KV-cache traits. Full transformer block extraction is deferred until a second native model exists, so GPT-2 learned-position blocks and future RoPE-based Qwen/Llama blocks do not get forced through the wrong abstraction.

See `docs/model-runtime.md` for shared autoregressive runtime notes and examples.

### Whisper native runtime status

The `whisper` command currently supports asset preparation and the native audio preprocessing path. It downloads or checks these Hugging Face files: `config.json`, `tokenizer.json`, `preprocessor_config.json`, and `model.safetensors`.

Prepare the default `tiny.en` assets and print resolved metadata:

```bash
./target/release/puppygrad whisper \
  --size tiny.en \
  --download \
  --print-config
```

The default local directory is `models/whisper-tiny.en`. Use `--model-dir`, `--model-id`, and `--revision` to select a different checkpoint or location.

The smoke-test command for the intended first transcription path is:

```bash
./target/release/puppygrad whisper \
  --audio tests/data/audio/jfk_16khz_mono.wav \
  --size tiny.en \
  --download \
  --task transcribe \
  --language en \
  --no-timestamps \
  --stats
```

When `--max-new-tokens` is omitted, Whisper decodes until EOS or the remaining decoder text context is full. Pass `--max-new-tokens N` only when you want to cap a run for a shorter smoke test.

Use `--audio -` to read 16 kHz PCM WAV bytes from stdin, for example `cat clip.wav | ./target/release/puppygrad whisper --audio - --size tiny.en --language en --no-timestamps`.

For segment metadata instead of plain text, pass `--output json`. `--output srt` and `--output vtt` emit segment-window subtitle timestamps by default; with `--timestamps`, Whisper timestamp tokens are decoded into segment timings. Audio longer than one 30-second Whisper window is split into consecutive windows; by default later segments may include previous segment text in the prompt. Pass `--no-condition-on-previous-text` to disable that. Use `--no-speech-threshold` to skip segments when the model's no-speech probability is high enough.

The native CPU path defaults to one worker thread for reproducibility. Pass `--threads N` to parallelize Whisper dense projections, convolution projections, final logits, and attention heads. Size presets provide default chunk sizes for `tiny.en` through `turbo`; `--print-config` includes the resolved Rust CPU tuning. `--quantized-weights` uses experimental row-wise int8 logits weights while keeping the hidden-state path in f32. `--backend gpu` is currently a typed hook that fails clearly until Whisper GPU kernels are implemented.

Whisper timing sweeps are available through:

```bash
./target/release/puppygrad experiment whisper \
  --audio tests/data/audio/jfk_16khz_mono.wav \
  --size tiny.en \
  --threads 4 \
  --max-new-tokens 8 \
  --runs 3
```

`autotune whisper` can rank max-new-token candidates for the current CPU reference path:

```bash
./target/release/puppygrad autotune whisper \
  --audio tests/data/audio/jfk_16khz_mono.wav \
  --size tiny.en \
  --max-new-tokens 1,2,4 \
  --runs 2
```

Known limitations: the command currently uses a straightforward CPU reference path with full-sequence decoder passes and no active KV-cache reuse, so long clips decode slowly. Quantization currently covers the logits projection only, and GPU execution is still TODO. By default the audio loader accepts PCM WAV; building with `--features audio-formats` also enables FLAC decoding. Arbitrary sample-rate inputs are decoded but still rejected by the Whisper command unless they are 16 kHz, so convert audio to 16 kHz mono PCM WAV for the default path.

The current supported Whisper size presets are `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, and `turbo`. Presets are used for model ids, default local directory names, approximate size metadata, and fallback architecture shape expectations; downloaded `config.json` and `preprocessor_config.json` are loaded and validated as the source of truth at runtime.

Whisper audio fixtures live in `tests/data/audio/`. See `tests/data/audio/README.md` for source URLs, conversion commands, formats, and SHA-256 checksums so the clips can be refreshed intentionally.

Transformer block sharing has been revisited now that GPT-2 and Whisper both have native paths. The code keeps them separate for now: GPT-2 is decoder-only with cached causal self-attention, while Whisper has an audio convolution/encoder stack plus decoder cross-attention. Shared CPU kernels remain in `src/models/cpu.rs`; a higher-level block abstraction should wait until a third model proves the common shape.

### Run GPT-2 small

First run downloads GPT-2 small assets into `models/gpt2`:

```bash
./target/release/puppygrad gpt2 \
  --download \
  --backend rust \
  --threads 4 \
  --stats \
  --prompt "Hello, my name is" \
  --max-new-tokens 20
```

After assets are downloaded, `--download` is optional:

```bash
./target/release/puppygrad gpt2 \
  --backend rust \
  --threads 4 \
  --stats \
  --prompt "The future of GPU compilers is" \
  --max-new-tokens 20
```

Use a different GPT-2-family checkpoint by giving both a model id and local directory:

```bash
./target/release/puppygrad gpt2 \
  --download \
  --model-id gpt2-medium \
  --model-dir models/gpt2-medium \
  --backend rust \
  --threads 4 \
  --prompt "Rust makes systems programming" \
  --max-new-tokens 20
```

The GPT-2 runtime is intentionally simple: CPU `f32` and no GPU kernels yet. The only backend today is `rust`; `--threads` controls puppygrad's own thread pool. GPT-2 runs print generated-token throughput to stderr after generation.

Greedy decoding is the default. For less repetitive text, enable sampling and repeat penalty:

```bash
./target/release/puppygrad gpt2 \
  --prompt "Hello, my name is" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  --repeat-last-n 128 \
  --seed 42
```

When `models/gpt2/puppygrad-tune.json` exists, the `gpt2` command loads it automatically. Explicit CLI flags override the saved config. Use `--no-tuning` to ignore the saved file, or `--tuning-file path/to/tune.json` to load a different file.

Pass `--stats` to print the full performance breakdown to stderr while streamed text stays on stdout. The current GPT-2 stats include model load time, tokenization time, prefill time, time to first token, decode time, average decode-token latency, and token/sec rates for prefill, decode, and total model tokens.

### Run GPT-2 experiments

Sweep backend settings and print averaged performance rows:

```bash
./target/release/puppygrad experiment gpt2 \
  --threads 1,2,4,8 \
  --max-new-tokens 16,32,64 \
  --runs 5 \
  --warmup-runs 1 \
  --prompt "The future of GPU compilers is"
```

Use `--format csv` or `--format json` when you want to plot results or compare runs outside the terminal.

### Autotune GPT-2 settings

Search candidate backend settings and print the fastest measured config:

```bash
./target/release/puppygrad autotune gpt2 \
  --threads 1,2,4,8,12,16,24,32 \
  --max-new-tokens 16 \
  --runs 2 \
  --warmup-runs 1 \
  --max-trials 48 \
  --prompt "The future of GPU compilers is"
```

The autotuner is generic internally: a target provides candidate configs, a trial runner, and a score. GPT-2 currently scores candidates by generated-token decode throughput.

By default, GPT-2 autotune saves the best config to `models/gpt2/puppygrad-tune.json`. Pass `--save-tuning path/to/file.json` to choose another location.

## Qwen Runtime Placeholder

```bash
./target/release/puppygrad qwen \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain RoPE in simple words." \
  --max-new-tokens 120
```

The `qwen` command is currently a stub. The previous external runtime was removed so model work can focus on low-level implementation inside this project.

The CLI still accepts local model paths for future native loading work:

```bash
./target/release/puppygrad qwen \
  --model-dir ./models/qwen2.5-0.5b-instruct \
  --prompt "Write a short Rust tip." \
  --max-new-tokens 120
```

## Run linear regression demo

```bash
./target/release/puppygrad demo-linear --steps 300 --lr 0.1
```

## Run matmul backward check

```bash
./target/release/puppygrad matmul-check
```

## Engine status

Implemented:

- Dynamic computation graph with reverse-mode autodiff.
- Scalar + vector operations: `add`, `sub`, `mul`, `relu`, `tanh`, `sum`, `mean`.
- 2D `matmul` forward and backward.
- Scalar broadcast support in binary ops.
- SGD parameter update helpers (`zero_grad`, `step`).

Current limits:

- CPU only.
- No advanced broadcasting rules beyond scalar broadcast.
- No kernel fusion or SIMD tuning yet.
- The GPT-2 runtime is a reference implementation, not optimized tensor infrastructure.
