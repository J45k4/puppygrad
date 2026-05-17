# Puppygrad TODO

## Whisper Native Runtime Plan

### Phase 1: Model Shape And Assets

- [x] Add `src/models/whisper/` module boundary with `mod.rs`.
- [x] Add Whisper error/result types.
- [x] Add Whisper size enum for `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, and `turbo`.
- [x] Add size preset metadata: Hugging Face model id, default local directory name, English-only flag, approximate parameter count, approximate VRAM, and relative speed.
- [x] Add `WhisperConfig` architecture dimensions for audio encoder and text decoder.
- [x] Add config validation for nonzero dimensions and head divisibility.
- [x] Add Hugging Face asset paths for `config.json`, `tokenizer.json`, `preprocessor_config.json`, and `model.safetensors`.
- [x] Add audio fixture directory with committed 16 kHz mono WAV clips and source/checksum manifest for regenerating originals.
- [x] Wire a `whisper` CLI command that can prepare/download assets and print resolved model metadata.
- [x] Add README section for Whisper model sizes, asset preparation, and current implementation status.
- [x] Add fixture provenance note to README or developer docs so test clips can be refreshed intentionally.

### Phase 2: Config And Preprocessor Loading

- [x] Load Hugging Face Whisper `config.json` into `WhisperConfig`.
- [x] Load `preprocessor_config.json` into a typed audio preprocessing config.
- [x] Validate expected sample rate, chunk length, hop length, FFT size, feature count, and normalization settings.
- [x] Add tests that parse representative configs for `tiny`, `large-v3`, and `turbo`.
- [x] Decide whether model presets are fallback-only or checked against downloaded configs.
- [x] Add `puppygrad whisper --print-config` or equivalent debug output for loaded model/preprocessor dimensions.

### Phase 3: Tokenizer And Special Tokens

- [x] Load Whisper `tokenizer.json` with the existing `tokenizers` dependency.
- [x] Add a `WhisperTokenizer` wrapper implementing the shared token decoder trait.
- [x] Identify and expose special tokens: start-of-transcript, EOS, no-timestamps, translate, transcribe, language tokens, timestamp range, and no-speech token if present.
- [x] Build prompt prefix creation for transcription mode.
- [x] Build prompt prefix creation for translation mode.
- [x] Add language selection support for multilingual checkpoints.
- [x] Add tests for tokenization, decoding, and prompt-prefix token IDs.
- [x] Add tokenizer tests using downloaded tiny/tiny.en assets once asset preparation is wired.

### Phase 4: Audio Input And Log-Mel Features

- [x] Add audio file loading strategy. Support WAV first with `jfk_16khz_mono.wav` and `micro_machines_16khz_mono.wav`; keep FLAC support out of the MVP unless a dependency is deliberately added.
- [x] Decode mono/stereo PCM WAV into normalized `f32` samples.
- [x] Validate `jfk_16khz_mono.wav` loads as 16 kHz mono PCM and has 176,000 samples.
- [x] Validate `micro_machines_16khz_mono.wav` loads as 16 kHz mono PCM and is near one full Whisper 30s window.
- [x] Add stereo/downmix tests using generated in-memory WAV data instead of committing a large stereo fixture.
- [x] Add resampling path if input sample rate is not Whisper's expected sample rate, or explicitly reject non-16 kHz WAVs until resampling is implemented.
- [x] Implement padding/trimming to Whisper chunk length.
- [x] Implement STFT.
- [x] Implement mel filterbank generation/loading.
- [x] Implement log-mel spectrogram extraction.
- [x] Match Whisper feature normalization semantics.
- [x] Add deterministic log-mel feature snapshot tests for `jfk_16khz_mono.wav`.
- [x] Add deterministic log-mel feature snapshot tests for `micro_machines_16khz_mono.wav`.
- [x] Compare log-mel output for `jfk_16khz_mono.wav` against Python Whisper or another trusted reference and record tolerance.

### Phase 5: Weight Loading

- [x] Define `WhisperWeights` with audio encoder, text decoder, cross-attention, embeddings, layer norms, and final projection.
- [x] Map Hugging Face safetensors names to internal weight structs.
- [x] Load all encoder block weights.
- [x] Load all decoder block weights, including self-attention and cross-attention.
- [x] Load token embeddings and positional embeddings.
- [x] Support tied output projection if the checkpoint uses token embeddings for logits.
- [x] Validate every tensor shape against `WhisperConfig`.
- [x] Add focused tests with tiny synthetic tensors to catch name/shape mistakes.
- [x] Add a tiny/tiny.en real checkpoint load smoke test that validates tensor names and shapes without running inference.

### Phase 6: Encoder Forward Pass

- [x] Implement audio input projection/convolution stack.
- [x] Add encoder positional embeddings.
- [x] Implement encoder self-attention.
- [x] Implement encoder MLP.
- [x] Implement encoder layer norm/residual flow.
- [x] Return encoded audio memory in a layout suitable for decoder cross-attention.
- [x] Add operation-level profile buckets for audio projection, encoder attention, encoder MLP, and encoder layer norms.
- [x] Validate encoder output shape for each size preset.
- [x] Run encoder shape smoke test on `jfk_16khz_mono.wav` with `tiny.en`.
- [x] Compare a small slice of encoder output against a trusted reference for `jfk_16khz_mono.wav`.

### Phase 7: Decoder Forward Pass

- [x] Define decoder KV cache for text self-attention.
- [x] Decide whether cross-attention K/V over encoded audio should be precomputed and cached.
- [x] Implement decoder token and positional embeddings.
- [x] Implement masked decoder self-attention.
- [x] Implement decoder cross-attention over encoder output.
- [x] Implement decoder MLP.
- [x] Implement decoder layer norm/residual flow.
- [x] Implement final logits projection.
- [x] Add decoder-only unit tests with synthetic weights and small dimensions.
- [x] Compare first-step logits for `tiny.en` on `jfk_16khz_mono.wav` against a trusted reference.

### Phase 8: Conditional Autoregressive Generation

- [x] Add a shared conditional autoregressive trait where generation depends on `condition + previous tokens`.
- [x] Make Whisper's condition type be encoded audio memory.
- [x] Reuse `TextGenerationConfig`, `LogitsSampler`, token streaming, and generic generation stats.
- [x] Implement greedy decoding first.
- [x] Add temperature/top-k/top-p sampling after greedy correctness is stable.
- [x] Add EOS stopping.
- [x] Add timestamp-token suppression or enabling flags.
- [x] Add no-timestamps mode for plain transcription.
- [x] Add no-speech handling once the decoder exposes the needed logits/probabilities.
- [x] Add an end-to-end greedy decode smoke test for `jfk_16khz_mono.wav` with `tiny.en`.
- [x] Add an end-to-end near-30s decode smoke test for `micro_machines_16khz_mono.wav` with `tiny.en`.

### Phase 9: CLI MVP

- [x] Add `puppygrad whisper --audio path.wav --size tiny --download`.
- [x] Add `--model-dir`, `--model-id`, and `--revision`.
- [x] Add `--task transcribe|translate`.
- [x] Add `--language` for multilingual models.
- [x] Add `--timestamps` / `--no-timestamps`.
- [x] Reuse shared generation args where they make sense.
- [x] Add `--stats` for load, audio preprocessing, encode, prefill, decode, and total timings.
- [x] Print transcript to stdout and stats to stderr.
- [x] Confirm `puppygrad whisper --audio tests/data/audio/jfk_16khz_mono.wav --size tiny.en --download --task transcribe --language en --no-timestamps --stats` works from a clean checkout after assets download.
- [x] Confirm CLI errors clearly for unsupported audio formats, missing assets, invalid language, and unsupported timestamp mode.

### Phase 10: Long Audio And Segments

- [x] Split audio into Whisper-sized windows.
- [x] Carry previous text prompt between segments when useful.
- [x] Track segment start/end times.
- [x] Decode timestamp tokens into segment timings.
- [x] Add VAD/no-speech based segment skipping if model outputs support it.
- [x] Produce plain text output first.
- [x] Add optional JSON segment output.
- [x] Add optional SRT/VTT output later.

### Phase 11: Performance Work

- [x] Reuse existing CPU kernels where possible before adding Whisper-specific kernels.
- [x] Parallelize encoder layers and projections where safe.
- [x] Parallelize attention heads for encoder, decoder self-attention, and cross-attention.
- [x] Add persistent scratch buffers for encoder and decoder.
- [x] Add transposed dense weights at model construction.
- [x] Add model-size-specific chunk tuning.
- [x] Add `experiment whisper` for audio preprocessing, encode, and decode timings.
- [x] Add `autotune whisper` after the reference path is correct.

### Phase 12: Correctness And Compatibility

- [x] Compare tiny model logits against a reference implementation on `jfk_16khz_mono.wav`.
- [x] Compare generated transcript for `jfk_16khz_mono.wav`; expected text should contain "ask not what your country can do for you".
- [x] Compare generated transcript for `micro_machines_16khz_mono.wav`; use it as a longer-window smoke test rather than a strict word-for-word fixture at first.
- [x] Test `tiny.en` and `tiny` first.
- [x] Test `base`, `small`, `medium`, `large-v3`, and `turbo` after tiny is stable.
- [x] Document unsupported features clearly while runtime is incomplete.
- [x] Keep Whisper-specific architecture code separate until duplication with GPT-2 or future models proves a shared abstraction.

### Phase 13: Future Extensions

- [x] Add quantized weight path only after f32 correctness is stable.
- [x] Add GPU backend hooks after CPU reference behavior is reliable.
- [x] Consider broader audio format support through an optional feature flag.
- [x] Consider streaming microphone/audio input after file transcription works.
- [x] Revisit transformer block extraction once GPT-2 and Whisper both have working native paths.

## Completion Criteria

- [x] `cargo fmt --check`, `cargo test`, and `cargo clippy -- -D warnings` pass with Whisper code and fixtures present.
- [x] A clean checkout can prepare `tiny.en` assets using the documented `puppygrad whisper --download` flow.
- [x] `jfk_16khz_mono.wav` loads through the native WAV path as 16 kHz mono PCM with the expected duration/sample count.
- [x] Native log-mel features for `jfk_16khz_mono.wav` match a trusted Whisper reference within documented tolerance.
- [x] `tiny.en` `config.json`, `preprocessor_config.json`, `tokenizer.json`, and `model.safetensors` load into typed Puppygrad structs with full shape validation.
- [x] Encoder forward pass runs on `jfk_16khz_mono.wav` and returns the expected encoded-audio shape.
- [x] Decoder first-token logits for `jfk_16khz_mono.wav` match a trusted reference within documented tolerance.
- [x] End-to-end greedy transcription for `jfk_16khz_mono.wav` produces recognizable JFK text containing "ask not what your country can do for you".
- [x] End-to-end transcription for `micro_machines_16khz_mono.wav` runs without crashing and produces non-empty text.
- [x] The CLI prints transcript text to stdout and timing/profile information to stderr when `--stats` is set.
- [x] Unsupported paths fail clearly: missing model files, missing audio, unsupported audio format, incompatible config, and conflicting timestamp flags.
- [x] README documents current Whisper support, exact smoke-test command, fixture sources, and known limitations.
