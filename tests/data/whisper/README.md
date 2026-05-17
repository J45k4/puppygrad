# Whisper Reference Fixtures

- `mel_filters.npz`
  - Source: `https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz`
  - Official OpenAI Whisper mel filterbank asset.

- `jfk_logmel_openai_ref_f32.bin`
  - Derived from `tests/data/audio/jfk_16khz_mono.wav` with a NumPy implementation of OpenAI Whisper's log-mel preprocessing semantics and the official `mel_80` filterbank above.
  - Format: little-endian contiguous `f32`, shape `[80, 3000]`, mel-major row order.
  - Regenerate intentionally with: center-pad 200 samples, Hann window with `sin(pi*n/400)^2`, `rfft`, power spectrum, drop the final frame, apply `mel_80`, `log10(max(x, 1e-10))`, clamp to `max - 8`, then normalize with `(x + 4) / 4`.
  - Native Puppygrad log-mel output is tested against this snapshot with max absolute tolerance `2e-3`.

- `jfk_tiny_en_first_logits_openai_ref_f32.bin`
  - Derived from OpenAI Whisper `tiny.en` on `tests/data/audio/jfk_16khz_mono.wav`.
  - Prompt tokens: `<|startoftranscript|>`, `<|transcribe|>`, `<|notimestamps|>`.
  - Format: little-endian contiguous `f32`, shape `[51864]`.
  - Native Puppygrad first-step logits were compared against this snapshot with max absolute difference `0.03593`, mean absolute difference `0.00875`, and matching argmax token `843`.

- `jfk_tiny_en_encoder_slice_openai_ref_f32.bin`
  - Derived from OpenAI Whisper `tiny.en` encoder output on `tests/data/audio/jfk_16khz_mono.wav`.
  - Format: first 64 little-endian contiguous `f32` values from encoder output shape `[1500, 384]`.
  - Native Puppygrad encoder output was compared against this slice with max absolute difference `0.00350` and mean absolute difference `0.00085`.
