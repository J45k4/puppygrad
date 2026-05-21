# Puppygrad TODO

## Goal: Piper ONNX To Audio

Build the shortest honest path from a downloaded Piper voice to a real synthesized audio clip:

1. Parse model weights from `model.onnx` directly in Rust.
2. Load those weights into typed Piper/VITS structs.
3. Run native inference and write a WAV clip.

Current local example voice:

- `models/piper/model.onnx`
- `models/piper/model.onnx.json`
- voice: `en_US-lessac-medium`
- ONNX size: about 60 MiB
- parsed initializers: 401
- initializer dtypes: 366 `FLOAT`, 34 `INT64`, 1 `INT32`
- total initializer elements: 15,650,556
- stored tensor bytes: 62,602,544 (59.70 MiB)
- largest tensor: `dec.ups.0.weight`, shape `[256, 128, 16]`
- dtype note: the formerly single `other` initializer is ONNX `INT32` (`/dp/flows.7/Constant_86_output_0`), a scalar graph constant rather than a learned VITS weight.

## 1. Parse Weights From ONNX

- [x] Add dependency-free ONNX/protobuf wire parser.
- [x] Parse `ModelProto.graph.initializer`.
- [x] Parse TensorProto names, dims, dtype, `raw_data`, `float_data`, and `int64_data`.
- [x] Decode `FLOAT` tensors from `raw_data`.
- [x] Decode `FLOAT` tensors from repeated `float_data`.
- [x] Expose initializer iteration and tensor summaries.
- [x] Run parser against `models/piper/model.onnx`.
- [x] Record initializer count, dtype summary, total size, and largest tensors.
- [x] Decode `INT64` tensors from `raw_data`.
- [x] Decode signed integer tensor fields correctly if needed.
- [x] Identify the single `other` dtype tensor and decide whether it matters for inference.
- [x] Add a debug command to print initializer names, dtypes, shapes, element counts, and byte sizes.
- [x] Save a full local initializer manifest for `en_US-lessac-medium`.
- [x] Add a local-only smoke test gated on `models/piper/model.onnx` existing.

## 2. Load Weights Into VITS

- [x] Keep Piper adapter code in `src/models/piper/`.
- [x] Keep reusable VITS architecture code in `src/models/vits/`.
- [x] Add `src/models/vits/weights.rs`.
- [x] Define typed structs for:
  - text encoder embedding and projection
  - text encoder attention layers
  - text encoder FFN layers
  - duration predictor
  - stochastic duration predictor / flows
  - residual coupling flow blocks
  - generator pre/post convs
  - generator upsample conv-transpose layers
  - generator ResBlock1/ResBlock2 layers
  - optional speaker embedding
- [x] Map `en_US-lessac-medium` ONNX initializer names to those structs.
- [x] Validate every required tensor rank and shape against config-derived dimensions.
- [x] Add clear errors for missing tensor, wrong dtype, wrong rank, wrong shape, and unsupported storage.
- [x] Handle single-speaker voices without speaker embedding.
- [x] Handle multi-speaker voices with speaker embedding and speaker id.
- [x] Add focused unit tests using handcrafted initializer stores.
- [x] Add a local real-model weight-loading smoke test.

## 3. Generate Audio Clip

- [x] Parse Piper voice config JSON.
- [x] Parse Piper phoneme id maps.
- [x] Accept raw phoneme ids in the `piper` CLI.
- [x] Write WAV output through the existing audio module.
- [x] Implement VITS text encoder forward pass.
- [x] Implement deterministic duration predictor forward pass from loaded weights.
- [x] Implement stochastic duration predictor reverse path, or bypass it safely if the voice/export permits.
- [x] Implement duration path expansion for prior mean/log-scale.
- [x] Implement residual coupling flow reverse pass from loaded weights.
- [x] Implement generator/vocoder forward pass from loaded weights.
- [x] Add deterministic RNG injection for reproducible inference tests.
- [x] Wire full Piper inference:
  - phoneme ids
  - optional speaker id
  - noise scale
  - length scale
  - duration noise scale
  - VITS forward path
  - mono `f32` waveform
- [x] Replace debug waveform synthesis in `puppygrad piper` with real VITS inference.
- [x] Generate a real WAV from a short phoneme-id input.
- [x] Add `--text` real synthesis through the platform TTS fallback while native eSpeak phonemization/VITS parity is unfinished.
- [x] Compare generated output shape/duration against Piper ONNX Runtime for one fixed input.
- [x] Document exact command for generating the first real clip.

## Done When

- [x] `models/piper/model.onnx` weights load into typed VITS structs.
- [x] `puppygrad piper --model-dir models/piper --phoneme-ids ... --out out.wav` produces real speech, not the debug tone.
- [x] The generated WAV is finite, non-empty, and has the sample rate from `model.onnx.json`.
- [x] `cargo fmt --check` passes.
- [x] Piper/VITS unit tests pass.
- [x] Full `cargo test` passes.
