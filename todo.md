# Puppygrad TODO

## Goal: Bark Native Rust Backend

Build a real Rust Bark inference path that can generate a WAV without calling Python,
Transformers, Torch, ONNX Runtime, or another model execution process.

Primary correctness objective:

- The Rust backend is not required to produce byte-identical final audio for
  every same input. The objective is a standalone Rust Bark path whose
  deterministic preprocessing, model staging, tensor layouts, logits, generated
  code shapes, and decoded audio properties are validated against Hugging Face
  Transformers closely enough to prove it is the same model path, not merely a
  plausible Bark-like implementation.
- Every comparison must use identical model assets, text prompt, voice
  preset/history prompt, seed, and generation settings.
- The Rust backend must be able to stand alone. Python Transformers is only an
  external reference fixture generator and CLI comparison backend; Rust model
  code must not import it, call it, shell out to it, or depend on it at runtime.
- Exact matches are required for deterministic preprocessing and layout
  decisions: tokenizer ids, semantic input ids, attention masks, codebook
  ordering, tensor shapes, and model input windows.
- Exact matches are required for semantic/coarse/fine token and code outputs
  only when generation is deterministic, such as greedy decoding or a fixed-seed
  sampling path that is intentionally made equivalent.
- If sampled generation or floating-point decoder math cannot be made
  bit-for-bit identical, prove equivalence with reference logits, valid-token
  masks, distributions, code shapes, waveform duration, waveform statistics,
  and short waveform slices within documented tolerances.
- Bit-for-bit identical final WAV output is not a blanket requirement unless we
  intentionally match every numerical and sampling detail needed to guarantee it.
- End-to-end acceptance compares Rust and Python Transformers outputs for the
  same prompt/settings through tokenizer ids, semantic/coarse/fine code shapes
  and deterministic values where applicable, EnCodec decode slices, waveform
  shape, sample rate, duration, finite samples, and bounded waveform statistics.

Target command:

```bash
./target/release/puppygrad bark \
  --model-dir models/bark-small \
  --backend rust \
  --text "hello from puppygrad" \
  --out /tmp/bark-rust.wav
```

Current Bark state:

- `src/models/bark/` parses Bark metadata and tokenizes text.
- `puppygrad bark --backend python-transformers --out ...` still uses the embedded Python Transformers bridge as a reference/backend option.
- `puppygrad bark --backend rust --out ...` enters the native Rust Bark runtime and requires converted `model.safetensors` weights.
- Native Rust weight loading plus semantic, coarse, fine, and EnCodec decoder scaffolding exists.
- Native Rust EnCodec waveform decoding has synthetic tests, but still needs real-checkpoint and Python reference parity validation before the Rust backend can be considered done.
- Scope note: the native backend is targeting `suno/bark-small` first; full-size Bark and arbitrary Bark variants stay out of scope until small works end-to-end.

## 1. Define The Rust Runtime Contract

- [x] Add `rust` to `BarkBackendArg`.
- [x] Keep the Rust runtime independent from the Python backend; do not call or import the Python path from Rust model code.
- [x] Keep `python-transformers` only as a CLI backend and developer reference for end-to-end comparison while the Rust path is validated.
- [x] Add a `BarkRuntimeOptions` struct for:
  - text
  - optional voice preset/history prompt
  - seed
  - semantic/coarse/fine temperatures
  - top-k/top-p
  - max semantic tokens
  - output WAV path
- [x] Add a public Bark runtime entry point under `src/models/bark/`, not in `main.rs`.
- [x] Keep CLI-only file handling and user-facing messages in `main.rs`.
- [x] Return generated mono `f32` samples plus sample rate from the Bark runtime.
- [x] Decide supported first target: `suno/bark-small` only.
- [x] Document that full-size Bark and arbitrary Bark variants are out of scope until small works.

## 2. Make Bark Assets Native-Friendly

- [x] Decide the native weight format:
  - preferred: converted `model.safetensors`
  - fallback: a minimal PyTorch checkpoint extractor/converter
- [x] Add Bark asset constants for native weights.
- [x] Keep `pytorch_model.bin` only for Python backend compatibility.
- [x] Add a conversion command or documented script:
  - input: Hugging Face Bark `pytorch_model.bin`
  - output: `model.safetensors`
  - output: optional manifest JSON with tensor names, dtypes, shapes, and byte sizes
- [x] Add asset validation that reports a clear error when `--backend rust` has only `pytorch_model.bin`.
- [x] Add local-only smoke test gates for `models/bark-small/model.safetensors`.
- [x] Record the exact expected tensor namespace for `suno/bark-small`.

## 3. Build Typed Bark Weight Loading

- [x] Add `src/models/bark/weights.rs`.
- [x] Define typed tensor structs for the semantic transformer.
- [x] Define typed tensor structs for the coarse acoustics transformer.
- [x] Define typed tensor structs for the fine acoustics transformer.
- [x] Define typed tensor structs for the EnCodec decoder.
- [x] Map Hugging Face tensor names into those typed structs.
- [x] Validate every required tensor:
  - [x] present
  - [x] `f32`
  - [x] expected rank
  - [x] expected shape
  - [x] no unused required layer groups
- [x] Support tied or separately stored token/output embeddings according to the checkpoint.
- [x] Add clear errors for missing tensor, wrong dtype, wrong rank, wrong shape, and unsupported checkpoint layout.
- [x] Add handcrafted unit tests for successful and failing weight loads.
- [x] Add a local real-checkpoint load test gated on native Bark weights existing.

## 4. Tokenizer Parity

- [x] Parse `tokenizer_config.json` and `special_tokens_map.json` instead of hard-coding all special tokens.
- [x] Match Hugging Face basic tokenization behavior needed by Bark.
- [x] Confirm lower-casing, accent handling, punctuation splitting, unknown-token behavior, and truncation behavior.
- [x] Add parity fixtures generated from Transformers for representative prompts:
  - short English
  - punctuation
  - numbers
  - mixed case
  - long text that truncates
  - unknown/non-ASCII text
- [x] Test raw token ids and semantic input ids after applying `text_encoding_offset`.
- [x] Keep the tokenizer dependency boundary explicit: either use the existing `tokenizers` crate or prove the local implementation matches the needed Bark tokenizer behavior.

## 5. Add Shared CPU Tensor Kernels Needed By Bark

- [x] Reuse existing CPU helpers where practical.
- [x] Add embedding lookup.
- [x] Add layer norm.
- [x] Add linear projection.
- [x] Add GELU or exact activation used by Bark transformer blocks.
- [x] Add multi-head causal self-attention.
- [x] Add attention mask handling.
- [x] Add KV cache support for autoregressive decoding.
- [x] Add residual block helpers.
- [x] Add logits processors:
  - temperature
  - top-k
  - top-p
  - invalid-token masking
  - EOS handling
- [x] Add deterministic RNG sampling from a seed.
- [x] Add small numerical tests for each kernel.
- [x] Add shape/error tests for invalid inputs.

## 6. Implement The Bark Transformer Block

- [x] Add `src/models/bark/transformer.rs`.
- [x] Implement the GPT-style block shape used by Bark semantic/coarse/fine models.
- [x] Support learned positional embeddings.
- [x] Support Bark-specific input vocab and output vocab sizes.
- [x] Support `block_size` constraints from config.
- [x] Support cached decoding for semantic and coarse generation.
- [x] Support full-context forward for fine generation if that matches Bark's fine path.
- [x] Add tests using tiny synthetic weights where expected logits are known.
- [x] Add debug hooks to dump shapes and selected logits when parity debugging is enabled.

## 7. Implement Semantic Token Generation

- [x] Add `src/models/bark/semantic.rs`.
- [x] Build semantic model inputs from tokenized text:
  - BarkProcessor-style text ids without `[CLS]`/`[SEP]`
  - text token offset
  - semantic infer token
  - padding as required by generation config
- [x] Run autoregressive semantic generation.
- [x] Stop on semantic EOS token.
- [x] Enforce `max_input_semantic_length`.
- [x] Enforce `max_new_tokens`.
- [x] Apply semantic temperature/top-k/top-p.
- [x] Mask invalid logits outside semantic vocabulary.
- [x] Return generated semantic token ids without text-offset ids.
- [x] Add parity fixture against Python Transformers for a fixed seed and short prompt.

## 8. Implement Coarse Acoustic Code Generation

- [x] Add `src/models/bark/coarse.rs`.
- [x] Convert semantic tokens into coarse model input format.
- [x] Implement Bark coarse codebook scheduling.
- [x] Respect:
  - `n_coarse_codebooks`
  - `coarse_rate_hz`
  - `semantic_rate_hz`
  - `max_coarse_history`
  - `max_coarse_input_length`
  - `sliding_window_len`
  - `coarse_infer_token`
  - `coarse_semantic_pad_token`
- [x] Apply coarse temperature/top-k/top-p.
- [x] Mask invalid logits outside the active coarse codebook range.
- [x] Return coarse acoustic codes with stable shape `[n_coarse_codebooks, frames]`.
- [x] Add parity fixture against Python Transformers intermediate coarse codes.

## 9. Implement Fine Acoustic Code Generation

- [x] Add `src/models/bark/fine.rs`.
- [x] Accept coarse codes as the first known codebooks.
- [x] Generate remaining fine codebooks.
- [x] Respect:
  - `n_fine_codebooks`
  - `max_fine_history_length`
  - `max_fine_input_length`
  - `n_codes_total`
  - `n_codes_given`
- [x] Confirm whether fine generation is sampled or greedy for Bark's default generation path.
- [x] Match Transformers behavior for codebook ordering and frame layout.
- [x] Return full EnCodec code matrix with shape `[n_fine_codebooks, frames]`.
- [x] Add parity fixture against Python Transformers intermediate fine codes.

## 10. Implement EnCodec Decode

- [x] Add `src/models/bark/encodec.rs`.
- [x] Load EnCodec quantizer/codebook embeddings.
- [x] Convert generated codebook ids into quantized latent vectors.
- [x] Implement the EnCodec decoder convolution stack used by Bark.
- [x] Implement required activations and normalization layers.
- [x] Implement transposed convolutions or reuse existing VITS conv-transpose helpers when shapes match.
- [x] Produce mono `f32` PCM samples.
- [x] Validate finite non-empty output.
- [x] Clamp or normalize only if Transformers does the same.
- [x] Add unit tests for decoder layer shapes.
- [x] Add local parity test comparing Rust decoded waveform with Python decoded waveform for fixed generated codes.

## 11. Voice Preset And History Prompt Support

- [x] Identify Bark voice preset asset names and layout used by Transformers.
- [x] Add Bark asset constants for prompt/history files if they are separate from the main checkpoint.
- [x] Parse semantic history prompts.
- [x] Parse coarse/fine acoustic history prompts.
- [x] Validate prompt token/code shapes.
- [x] Wire `--voice-preset` into the Rust backend.
- [x] Return a clear error when a requested preset is missing.
- [x] Add one local parity test for a known voice preset when assets exist.
  - `voice_preset_reference_fixtures_match_when_explicitly_enabled` runs for
    reference fixtures that include `voice_preset`; the local `models/bark-small`
    checkout currently has no speaker embedding assets, so no real preset
    fixture is checked in.

## 12. Wire The CLI

- [x] Add `BarkBackendArg::Rust`.
- [x] Make `--backend rust --out ...` call the native runtime.
- [x] Keep `--backend python-transformers` behavior unchanged.
- [x] Make `--print-config` and `--print-tokens` backend-independent.
- [x] Make `--download --backend rust` download or require native weights, not only Python weights.
- [x] Print backend, seed, sample rate, duration, and output path after successful Rust generation.
- [x] Ensure missing native assets produce actionable errors.
- [x] Ensure empty text fails before loading large weights.
- [x] Expose greedy/per-stage temperature/top-k/top-p/max semantic token CLI overrides.
- [x] Pass the same CLI generation overrides to the Rust backend and the `python-transformers` comparison backend.

## 13. Verification And Parity

- [x] Add a Python reference fixture generator for Bark intermediate outputs using Hugging Face Transformers.
- [x] Keep the Python reference generator outside the Rust runtime path.
- [x] Make the reference generator accept:
  - model directory
  - text prompt
  - optional voice preset
  - seed
  - semantic/coarse/fine sampling settings
  - output fixture path
- [x] Save reference fixture metadata:
  - model id or model directory
  - Transformers version
  - Torch version
  - seed
  - prompt text
  - voice preset
  - generation settings
  - sample rate
- [x] Save reference fixture revision metadata when available.
- [x] Capture tokenizer reference data:
  - [x] normalized text if exposed
  - [x] token strings
  - [x] raw token ids
  - [x] semantic input ids after Bark text offset
- [x] Capture semantic-stage reference data:
  - [x] semantic model input ids
  - [x] generated semantic tokens
  - [x] EOS position
  - [x] selected logits slice for first generation step
  - [x] top-k token ids/probabilities for first generation step
- [x] Capture coarse-stage reference data:
  - [x] coarse model input ids for the first window
  - [x] generated coarse codebooks
  - [x] coarse frame count
  - [x] selected logits slice for first coarse step
  - [x] top-k token ids/probabilities for first coarse step
- [x] Capture fine-stage reference data:
  - [x] fine model input code matrix
  - [x] generated fine codebooks
  - [x] full EnCodec code matrix
  - [x] selected logits slice for the first generated fine codebook step if exposed
- [x] Capture decoder reference data:
  - quantized latent shape
  - decoded waveform sample rate
  - decoded waveform sample count
  - first N waveform samples
  - waveform min/max/mean/RMS
- [x] Store only small generated fixtures in git:
  - tokenizer ids
  - short semantic token vector
  - short coarse/fine code slices
  - short waveform sample slice
  - metadata JSON
- [x] Keep large full-output fixtures local-only and ignored by git.
- [x] Add Rust fixture readers for JSON plus binary float/code slices.
  - [x] JSON fixture reader
  - [x] binary float/code slice reader
- [x] Add Rust parity tests for tokenizer output against Python fixtures.
- [x] Add Rust parity tests for semantic input construction before model execution.
- [x] Add Rust parity tests for semantic generated tokens when fixed-seed sampling is reproducible.
- [x] Add Rust parity tests for semantic logits slices with a numeric tolerance.
- [x] Add Rust parity tests for coarse input construction and codebook layout.
- [x] Add Rust parity tests for coarse generated codes when fixed-seed sampling is reproducible.
- [x] Add Rust parity tests for coarse logits slices with a numeric tolerance.
- [x] Add Rust parity tests for fine codebook layout.
- [x] Add Rust parity tests for fine generated codes when fixed-seed sampling is reproducible.
- [x] Add Rust parity tests for EnCodec decode from a fixed Python-generated code matrix.
- [x] Compare decoder waveform slices with an absolute/relative tolerance.
- [x] Compare final WAV-level properties:
  - sample rate
  - channel count
  - sample count
  - duration
  - finite samples
  - min/max/mean/RMS
- [x] Treat exact full waveform equality as non-goal unless the full math path is intentionally bit-for-bit matched.
- [x] Prefer deterministic greedy parity tests where possible to reduce sampling noise.
- [x] For sampled paths, compare fixed-seed token/code outputs first; if Torch and Rust RNG cannot match exactly, compare logits and distribution constraints instead.
- [x] Add an ignored/local end-to-end comparison script that runs both:
  - `puppygrad bark --backend python-transformers ...`
  - `puppygrad bark --backend rust ...`
  and compares sample rate, duration, finite output, and bounded waveform statistics.
- [x] Make the local backend comparison script accept the same greedy/sampling/max-token overrides as the Bark CLI.
- [x] Keep full model parity tests local-only and skipped when model assets are absent.
- [x] Add deterministic tests using fixed seed.
- [x] Compare exact token/code outputs where sampling is deterministic enough.
- [x] Compare waveform shape, sample rate, finiteness, duration, and tolerance-bounded sample slices.
- [x] Add CLI smoke test for metadata-only mode.
- [x] Add CLI smoke test for tokenizer mode.
- [x] Add local CLI smoke test for native WAV generation.
- [x] Syntax-check `scripts/bark_reference_fixture.py`.
- [x] Run `cargo fmt --check`.
- [x] Run `cargo test bark`.
- [x] Run full `cargo test` in the default no-hardware feature set.

## 14. Performance Pass After Correctness

- [x] Profile semantic generation.
- [x] Profile coarse generation.
- [x] Profile fine generation.
- [x] Profile EnCodec decode.
- [x] Avoid avoidable allocations in autoregressive loops.
- [x] Reuse KV-cache buffers.
- [x] Add optional thread-pool use only after single-thread correctness is established.
- [x] Document expected CPU runtime for a short prompt on the development machine.
  - Exact Done When command on this machine:
    `target/release/puppygrad bark --backend rust --text "hello" --out /tmp/bark-rust.wav`
    wrote 2.093s of mono 24 kHz audio in 65.24s wall time; stage timings
    were semantic 3.022s, coarse 12.793s, fine 39.678s, EnCodec 5.704s.
  - Current measured constrained smoke command on this machine:
    `target/release/puppygrad bark --model-dir models/bark-small --backend rust --text hello --out /tmp/bark-rust-short.wav --greedy --max-semantic-tokens 1 --threads 8`
    wrote 0.013s of mono 24 kHz audio in 54.78s wall time; stage timings were
    semantic 473.34ms, coarse 495.35ms, fine 49.731s, EnCodec 37.92ms.
  - Current local Python Transformers vs Rust comparison:
    `.venv-bark/bin/python scripts/bark_compare_backends.py --binary target/release/puppygrad --python .venv-bark/bin/python --model-dir models/bark-small --text hello --seed 299792458 --greedy --max-semantic-tokens 1 --threads 8`
    passes with identical 24 kHz mono, 320-sample output and matching waveform
    statistics for the deterministic fixture.

## Bark Done When

- [x] `puppygrad bark --backend rust --text "hello" --out /tmp/bark-rust.wav` writes a real speech WAV.
- [x] The Rust path does not execute Python, Torch, Transformers, ONNX Runtime, or another model process.
- [x] The WAV is finite, non-empty, mono, and uses the Bark generation sample rate.
- [x] Tokenizer ids match the Python reference fixtures.
- [x] For the same model assets, prompt, voice preset, seed, and generation settings, semantic, coarse, and fine generated codes match the Python Transformers reference only for deterministic generation paths; sampled paths have documented logits/distribution/shape tolerances instead.
- [x] EnCodec decode matches the Python reference closely enough for fixed generated codes.
- [x] A local end-to-end Python Transformers vs Rust comparison passes for the same prompt, seed, voice preset/history prompt, and generation settings.
- [x] Missing native weights produce a clear error with the expected file path and conversion instructions.
- [x] `cargo fmt --check` passes.
- [x] Bark unit tests pass.
- [x] Full `cargo test` passes in an environment with required system libraries.
