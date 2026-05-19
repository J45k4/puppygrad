# Puppygrad

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
| ResNet | Working MVP | Rust reference | Loads torchvision-origin ResNet-18 safetensors, folds Conv+BatchNorm at load time, preprocesses RGB images, and prints ImageNet top-k classes. |
| Qwen | Stub | None yet | CLI placeholder for future native loading/runtime work. |

Model assets are stored under the project-root `models/` directory, which is ignored by git. Rust source lives under `src/models/` and is tracked. GPT-2-specific code is organized under `src/models/gpt2/`, with the current Rust reference implementation in `src/models/gpt2/rust.rs`.

Shared model runtime code is intentionally limited to pieces that already have clear cross-model shape: generation CLI args and sampling config, token streaming, generation stats, asset/config loading, safetensors access, CPU math kernels, and minimal autoregressive/KV-cache traits. Full transformer block extraction is deferred until a second native model exists, so GPT-2 learned-position blocks and future RoPE-based Qwen/Llama blocks do not get forced through the wrong abstraction.

See `docs/model-runtime.md` for shared autoregressive runtime notes and examples.

### ResNet native runtime status

The `resnet` command currently supports ImageNet ResNet-18 through the Rust CPU reference path. The default asset directory is `models/resnet18`; `--download` fetches `timm/resnet18.tv_in1k` safetensors plus ImageNet labels. The checkpoint uses PyTorch-style keys such as `conv1.weight`, `bn1.*`, `layerN.B.convM.weight`, downsample projection keys, and `fc.*`.

Prepare assets and classify an image:

```bash
./target/release/puppygrad resnet \
  --download \
  --image tests/data/images/example.jpg \
  --top-k 5
```

After assets are present, `--download` is optional:

```bash
./target/release/puppygrad resnet \
  --image image.jpg \
  --top-k 5
```

Preprocessing decodes common RGB image files, resizes the shortest side to 256 pixels with bilinear filtering, center-crops to 224 x 224, converts to normalized CHW `[3, 224, 224]` data, and applies ImageNet mean/std normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

The runtime includes reusable vision pieces under `src/vision/`: image loading, HWC/CHW/NCHW layout helpers, resize, center crop, normalization, Conv2D, ReLU, pooling, residual add, linear classifier, softmax, and top-k helpers. CLIP and ViT can reuse the image loading/resize/crop/normalize path and classifier top-k patterns. YOLO can reuse image loading, the CNN kernels, activations, BatchNorm folding patterns, and later add letterbox and detection postprocessing.

Known limitations: CPU reference path only, ResNet-18 only, scalar convolution kernels, no GPU/backend dispatch, no object detection, and `--threads` is accepted for CLI compatibility but not used by the current ResNet path.

## Audio utilities

Puppygrad includes a shared audio CLI for microphone discovery, fixed-duration recording, and PCM WAV inspection. These commands do not load Whisper assets.

List input devices:

```bash
./target/release/puppygrad audio list-input-devices
```

The output is tab-separated as `index`, `name`, and an optional `default` marker. Device indices are intended for quick CLI selection and can change after reconnects, OS updates, or reboots.

Record a 3-second WAV from the OS default input device:

```bash
./target/release/puppygrad audio record \
  --seconds 3 \
  --out /tmp/puppygrad-mic.wav
```

Record from a selected input device:

```bash
./target/release/puppygrad audio record \
  --input-device 0 \
  --seconds 3 \
  --out /tmp/puppygrad-mic.wav
```

Inspect a WAV file:

```bash
./target/release/puppygrad audio inspect /tmp/puppygrad-mic.wav
```

`audio inspect` reports the sample rate, channel count, duration, sample count, and PCM WAV format summary. `audio record` writes mono 16-bit PCM WAV while preserving the input device sample rate.

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

Pass `--stream-raw-tokens` to stream raw decoder token events instead of the final transcript. Rows are tab-separated as `segment`, `phase`, `token_id`, `raw_token`; `phase=prompt` includes Whisper control tags such as `<|startoftranscript|>` and `phase=generated` includes model-produced tokens.

Whisper microphone mode listens continuously by default. Chunk mode records non-overlapping chunks and commits each decoded chunk as transcript text:

```bash
./target/release/puppygrad whisper \
  --mic \
  --chunk-seconds 4 \
  --size tiny.en \
  --language en
```

Use `--once` to preserve the old single-chunk smoke-test behavior:

```bash
./target/release/puppygrad whisper \
  --mic \
  --once \
  --chunk-seconds 4 \
  --max-new-tokens 64
```

Rolling mode keeps a moving audio window and emits editable partial transcript events while speech is ongoing. It commits text after silence or after the same normalized partial repeats enough times:

```bash
./target/release/puppygrad whisper \
  --mic \
  --mic-mode rolling \
  --window-seconds 8 \
  --partial-interval-ms 1000 \
  --max-new-tokens 64
```

Use `--input-device N` with `--mic` to select a device by the index shown by `audio list-input-devices`. `--audio` and `--mic` are mutually exclusive. Microphone input is captured on a continuous CPAL stream, queued independently from decode, downmixed, and linearly resampled to Whisper's 16 kHz mono input before log-mel preprocessing. `--max-queued-chunks N` controls the bounded capture queue; `--drop-policy oldest|newest|block` chooses overflow behavior, with `oldest` as the default. Queue overflow warnings and realtime stats go to stderr.

Rolling commit controls:

```bash
./target/release/puppygrad whisper \
  --mic \
  --mic-mode rolling \
  --commit-on-silence true \
  --silence-ms 900 \
  --silence-threshold 0.01
```

For downstream LLM input, prefer newline-delimited event JSON and consume only `commit` events as durable transcript turns. `partial` events are previews and may be replaced:

```bash
./target/release/puppygrad whisper \
  --mic \
  --mic-mode rolling \
  --output events-json \
  --window-seconds 8 \
  --partial-interval-ms 1000 \
  --max-new-tokens 64
```

Text output remains human-friendly: committed text is printed normally, and rolling partial text is updated in place without duplicating the active partial. `--output events-json` writes newline-delimited events such as `partial`, `commit`, `silence`, `warning`, and `raw_token` to stdout so status logs can stay on stderr.

Raw-token streaming works in chunk and rolling microphone modes:

```bash
./target/release/puppygrad whisper \
  --mic \
  --input-device 0 \
  --chunk-seconds 4 \
  --stream-raw-tokens
```

```bash
./target/release/puppygrad whisper \
  --mic \
  --mic-mode rolling \
  --output events-json \
  --stream-raw-tokens
```

Mic mode uses a bounded parallel CPU default. Pass `--threads N` and `--max-new-tokens N` to override the realtime defaults. A practical starting point for local LLM input is `--mic-mode rolling --window-seconds 8 --partial-interval-ms 1000 --max-new-tokens 64 --output events-json`.

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

Known limitations: the command currently uses a straightforward CPU reference path with full-sequence decoder passes and no active KV-cache reuse, so long clips and rolling windows can decode slowly. Rolling mode has a cheap RMS silence gate plus Whisper no-speech probability support, but it is still a lightweight realtime heuristic rather than polished VAD. Log-mel and encoder-output reuse for overlapping rolling windows are future optimizations. A backend-neutral job/result path is in place so a future scheduler can add micro-batching, but current microphone decode still runs one window at a time because single-mic latency matters more than throughput. CPU Whisper latency can be high on larger checkpoints. Quantization currently covers the logits projection only, and GPU execution is still TODO. By default the file audio loader accepts PCM WAV; building with `--features audio-formats` also enables FLAC decoding. Arbitrary sample-rate file inputs are decoded but still rejected by the Whisper file path unless they are 16 kHz, so convert files to 16 kHz mono PCM WAV for the default path.

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

Pass `--stream-raw-tokens` to stream GPT-2 token events instead of decoded text. Rows are tab-separated as `phase`, `token_id`, `raw_token`, with `phase=prompt` for prompt tokens and `phase=generated` for model-produced tokens.

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
