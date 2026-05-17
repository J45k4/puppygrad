# Puppygrad TODO

## Shared Audio And Whisper Mic Support Plan

### Phase 1: Audio Module Boundary

- [x] Add a shared audio module outside `src/models/whisper/`, for example `src/audio/` or `src/runtime/audio/`.
- [x] Keep OS/device audio capture, device listing, sample conversion, resampling, and WAV writing in the shared audio module.
- [x] Keep Whisper-specific log-mel preprocessing and model execution in `src/models/whisper/`.
- [x] Define a shared `AudioError` / `AudioResult` type so CLI and models do not need to depend on Whisper errors for generic audio operations.
- [x] Define a shared `PcmAudio` or compatible audio buffer type with sample rate, channel count/mono policy, and normalized `f32` samples.

### Phase 2: Audio CLI Namespace

- [x] Add a top-level `audio` CLI command group.
- [x] Add `puppygrad audio list-input-devices`.
- [x] Add `puppygrad audio record --seconds N --out clip.wav`.
- [x] Add `puppygrad audio record --input-device N --seconds N --out clip.wav`.
- [x] Add `puppygrad audio inspect path.wav` for sample rate, channel count, duration, sample count, and format summary.
- [x] Make shared audio commands independent from Whisper assets or model loading.

### Phase 3: Microphone Device Discovery

- [x] Add `cpal` or another deliberate cross-platform audio input dependency.
- [x] Enumerate input devices with stable display output.
- [x] Mark the OS default input device in `audio list-input-devices`.
- [x] Support default-device capture when no explicit input device is passed.
- [x] Support optional input-device index for quick CLI selection.
- [x] Document that device indices can change after reconnects/reboots.
- [x] Consider name matching later, for example `--input-device-name "MacBook"`.

### Phase 4: Microphone Capture

- [x] Implement blocking capture for a fixed duration.
- [x] Convert supported input sample formats to normalized `f32`.
- [x] Downmix multi-channel input to mono.
- [x] Preserve the original device sample rate in the captured buffer.
- [x] Add backpressure/buffer overflow handling with clear errors.
- [x] Make `audio record` write a valid PCM WAV file.
- [x] Add a short manual smoke-test recipe for recording and inspecting a clip.

### Phase 5: Resampling And Audio Normalization

- [x] Add a built-in linear resampler for the MVP, or choose a dedicated resampling crate if quality is worth the dependency.
- [x] Convert arbitrary mic sample rates such as 44.1 kHz and 48 kHz to Whisper's 16 kHz mono input.
- [x] Add unit tests for downmixing.
- [x] Add unit tests for sample format conversion.
- [x] Add unit tests for resampler output length and basic waveform shape.
- [x] Reuse the same normalization path for `audio record`, `audio inspect`, and Whisper mic input where practical.

### Phase 6: Whisper Input Modes

- [x] Add `--mic` to `puppygrad whisper`.
- [x] Make `--audio` and `--mic` mutually exclusive.
- [x] Add `--input-device N` for `puppygrad whisper --mic`.
- [x] Keep existing file and stdin modes:
  - `puppygrad whisper --audio clip.wav`
  - `puppygrad whisper --audio -`
- [x] Ensure model/backend/generation options continue to work with both file and mic input.
- [x] Fail clearly if microphone capture is requested on a platform/device setup that is unavailable.

### Phase 7: Live Chunking

- [x] Add `--chunk-seconds N` for mic mode.
- [x] Default mic chunks to a short value such as 5 seconds.
- [x] Convert each captured chunk to 16 kHz mono before Whisper preprocessing.
- [x] Run existing Whisper log-mel, encoder, and decoder paths per chunk.
- [x] Print chunk-level transcripts as each chunk finishes.
- [x] Track chunk start/end wall-clock times for future JSON/SRT/VTT support.
- [x] Defer overlap and rolling-context behavior until the basic mic path works.

### Phase 8: Streaming Output

- [x] Reuse `--stream-raw-tokens` for mic chunks.
- [x] Emit raw prompt/control tokens and generated tokens for each mic chunk.
- [x] Keep transcript text output as the default when raw streaming is disabled.
- [x] Add text streaming later if useful, separate from raw-token streaming.
- [x] Make stdout/stderr behavior consistent: transcripts/tokens on stdout, stats/status on stderr.

### Phase 9: Silence And Usability

- [x] Reuse Whisper no-speech probability for optional silence skipping.
- [x] Add `--no-speech-threshold` support for mic chunks.
- [x] Consider a simple RMS gate before running Whisper on silent chunks.
- [x] Avoid printing empty transcript lines by default.
- [x] Add clear status output for recording start/stop only on stderr.

### Phase 10: Tests And Manual Verification

- [x] Keep file-based Whisper tests as deterministic correctness tests.
- [x] Unit test shared audio conversion/downmix/resampling without requiring a real microphone.
- [x] Add `audio inspect` tests using existing 16 kHz WAV fixtures.
- [x] Add manual test notes for `audio list-input-devices`.
- [x] Add manual test notes for `audio record --seconds 3 --out /tmp/puppygrad-mic.wav`.
- [x] Add manual test notes for `whisper --mic --chunk-seconds 5 --stream-raw-tokens`.
- [x] Avoid making CI depend on an actual microphone.

### Phase 11: Documentation

- [x] Document `puppygrad audio list-input-devices`.
- [x] Document `puppygrad audio record`.
- [x] Document `puppygrad audio inspect`.
- [x] Document `puppygrad whisper --mic`.
- [x] Document default-device behavior and optional input-device index.
- [x] Document that mic audio is resampled/downmixed to 16 kHz mono for Whisper.
- [x] Document known limitations: chunk-level output first, no overlap/VAD polish initially, CPU Whisper latency.

## Completion Criteria

- [x] `cargo fmt --check` passes.
- [x] `cargo check` passes.
- [x] Shared audio unit tests pass without requiring a microphone.
- [x] `puppygrad audio list-input-devices` prints available input devices and marks the default device.
- [x] `puppygrad audio record --seconds 3 --out /tmp/puppygrad-mic.wav` records a playable WAV from the default microphone.
- [x] `puppygrad audio record --input-device N --seconds 3 --out /tmp/puppygrad-mic.wav` records from a selected device.
- [x] `puppygrad audio inspect /tmp/puppygrad-mic.wav` reports sample rate, channel count, duration, and sample count.
- [x] `puppygrad whisper --mic --chunk-seconds 5 --size tiny.en --language en --no-timestamps` records from the default mic and prints a transcript chunk.
- [x] `puppygrad whisper --mic --input-device N --chunk-seconds 5 --stream-raw-tokens` prints raw token events with prompt/control tags and generated tokens.
- [x] Existing Whisper file mode still works with `--audio tests/data/audio/jfk_16khz_mono.wav`.
- [x] README documents the audio commands, mic mode, and current limitations.
