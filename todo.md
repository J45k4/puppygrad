# Puppygrad TODO

## Realtime Rolling Whisper Plan

Goal: make microphone transcription feel responsive enough for downstream LLM use. The runtime should continuously capture audio, emit editable partial transcripts, commit stable speech turns on silence or stability, and keep the decode pipeline shaped so batching can be added later.

### Phase 1: Mic Mode Semantics

- [x] Change `puppygrad whisper --mic` from one-shot recording to continuous listening until Ctrl-C.
- [x] Add `--once` to preserve the current single-chunk smoke-test behavior.
- [x] Lower the mic default chunk/window behavior so the default feels interactive.
- [x] Remove the surprising mic default of 2 generated tokens, or replace it with a documented realtime default such as 64.
- [x] Keep file mode behavior unchanged for `--audio path.wav` and `--audio -`.
- [x] Document that mic mode is continuous by default and `--once` is for tests.

### Phase 2: Capture Pipeline

- [x] Replace blocking `record_input_device(duration)` usage in Whisper mic mode with a continuous capture stream.
- [x] Keep audio capture independent from Whisper decode speed.
- [x] Push captured mono/resampled audio into a bounded buffer or chunk queue.
- [x] Add a clear queue overflow policy: warn and drop oldest chunk first.
- [x] Ensure Ctrl-C / shutdown closes the audio stream cleanly.
- [x] Keep `audio record --seconds N` as a blocking fixed-duration utility command.

### Phase 3: Rolling Decode Mode

- [x] Add mic decode modes:
  - `--mic-mode chunks` for simple non-overlapping chunks.
  - `--mic-mode rolling` for responsive partial transcription.
- [x] Make `rolling` the preferred mode for realtime LLM use once stable.
- [x] Add `--window-seconds N` for the rolling audio window, default around 6-8 seconds.
- [x] Add `--partial-interval-ms N`, default around 750-1000 ms.
- [x] Keep `--chunk-seconds N` for simple chunk mode, default around 3-4 seconds.
- [x] Validate that window/interval/chunk values are positive and sane.

### Phase 4: Partial And Commit Events

- [x] Introduce a transcription event type:
  - `Partial { text }`
  - `Commit { text }`
  - `RawToken { phase, token_id, token }`
  - `Silence`
  - `Warning { message }`
- [x] In rolling mode, repeatedly decode the current rolling window and emit `Partial` when text changes.
- [x] Keep only one visible partial active in terminal text mode.
- [x] Commit text on silence, stable repeated decode, or explicit window boundary.
- [x] After commit, advance or clear the rolling buffer so committed text is not repeatedly re-emitted.
- [x] Keep committed text suitable for downstream LLM input.

### Phase 5: Silence And Stability

- [x] Add a cheap RMS silence gate before invoking Whisper.
- [x] Reuse Whisper no-speech probability when available.
- [x] Add `--commit-on-silence`, enabled by default in rolling mic mode.
- [x] Add `--silence-ms N`, default around 700-1200 ms.
- [x] Add `--silence-threshold N` for RMS gate tuning.
- [x] Add stable-text commit heuristic: if normalized partial text is unchanged for N decodes, commit it.
- [x] Avoid committing empty or whitespace-only text.

### Phase 6: Output Formats For Realtime

- [x] Keep plain text output as human-friendly terminal output.
- [x] Add an event output format for machines, likely newline-delimited JSON:
  - `--output events-json`
- [x] Make raw-token streaming work inside both chunk and rolling mic modes.
- [x] In terminal text mode, print committed text normally and update partial text without duplicating it.
- [x] Keep stats/status/warnings on stderr.
- [x] Ensure downstream tools can read committed transcript events from stdout without parsing status logs.

### Phase 7: Batching-Ready Job Model

- [x] Define a backend-neutral transcription job:
  - stream id
  - window start/end time
  - samples at 16 kHz mono
  - purpose: partial or final
  - generation settings
- [x] Define a decode result:
  - job id
  - decoded text
  - generated token ids
  - no-speech probability
  - timing/profile data
- [x] Route both chunk mode and rolling mode through the same job/result path.
- [x] Keep the first backend implementation as `decode_one(job)`.
- [x] Design the trait boundary so a future backend can add `decode_batch(&[job])`.
- [x] Avoid CLI-specific logic inside the model decode path.

### Phase 8: Performance And Latency Controls

- [x] Track capture lag, queue depth, decode time, and end-to-end partial latency.
- [x] Print realtime stats when `--stats` is enabled.
- [x] Add warning when decode falls behind realtime.
- [x] Add `--max-queued-chunks N`, default small such as 2.
- [x] Add `--drop-policy oldest|newest|block`, default `oldest`.
- [x] Reuse log-mel frames for overlapping rolling windows when practical.
- [x] Keep encoder-output reuse as a later optimization after correctness.

### Phase 9: Future Batching

- [x] Add a decode scheduler interface that can choose between immediate decode and micro-batching.
- [x] Add `--micro-batch-size N` later, default 1.
- [x] Add `--micro-batch-timeout-ms N` later to cap batching latency.
- [x] Support batching across multiple pending rolling windows if decode falls behind.
- [x] Support batching across multiple streams/users later.
- [x] Keep batching optional because single-mic realtime latency matters more than throughput.

### Phase 10: Tests And Manual Verification

- [x] Unit test rolling buffer append/slice/advance behavior.
- [x] Unit test RMS silence detection.
- [x] Unit test partial-to-commit state transitions.
- [x] Unit test duplicate partial suppression.
- [x] Unit test job/result scheduler plumbing without microphone access.
- [x] Keep microphone tests as manual smoke tests.
- [x] Verify `--once` still records one chunk and exits.
- [x] Verify default `--mic` continues until interrupted.
- [x] Verify raw-token streaming still prints prompt/control and generated tokens.

### Phase 11: Documentation

- [x] Update README to describe continuous mic mode.
- [x] Document `--once`.
- [x] Document `--mic-mode chunks|rolling`.
- [x] Document `--window-seconds`, `--partial-interval-ms`, and `--chunk-seconds`.
- [x] Document partial versus committed transcript behavior.
- [x] Document recommended settings for LLM input.
- [x] Document current CPU latency limitations and the future batching path.

## Completion Criteria

- [x] `cargo fmt --check` passes.
- [x] `cargo check` passes.
- [x] Realtime pipeline unit tests pass without requiring a microphone.
- [x] `puppygrad whisper --mic --once --chunk-seconds 4 --max-new-tokens 64` records one chunk, transcribes it, and exits.
- [x] `puppygrad whisper --mic --chunk-seconds 4 --max-new-tokens 64` records and transcribes chunks continuously until Ctrl-C.
- [x] `puppygrad whisper --mic --mic-mode rolling --window-seconds 8 --partial-interval-ms 1000` emits partial updates while speech is ongoing.
- [x] Rolling mode emits committed text after silence and does not repeatedly resend the same committed text.
- [x] `--output events-json` emits machine-readable partial/commit events on stdout.
- [x] `--stream-raw-tokens` works in mic chunk mode and rolling mode.
- [x] Existing file transcription still works with `--audio tests/data/audio/jfk_16khz_mono.wav`.
- [x] README documents realtime mic usage, commit semantics, and limitations.
