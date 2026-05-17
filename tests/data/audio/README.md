# Audio Fixtures

These clips are used for Whisper runtime development and tests.

## Committed Files

- `jfk_16khz_mono.wav`
  - Derived from OpenAI Whisper's `jfk.flac` fixture with `ffmpeg -ac 1 -ar 16000`.
  - Intended first end-to-end Whisper smoke fixture.
  - Format: WAV PCM s16le, 16 kHz, mono, 11.0s.
  - SHA-256: `4d968ac99a1d0d4bc42ae8dd1552f4235e7b952a8121b39797f3dc2ca16123e9`

- `micro_machines_16khz_mono.wav`
  - Derived from OpenAI Whisper's `micro-machines.wav` demo clip with `ffmpeg -ac 1 -ar 16000`.
  - Useful for a near-30s Whisper window test.
  - Format: WAV PCM s16le, 16 kHz, mono, 29.888375s.
  - SHA-256: `deb2a7ca2de6eb1fbd7397bed2dc1f230a2b07aa33189f02d29536b8248378f2`

## Source Files Not Committed

The original downloads are intentionally not committed to keep the repository smaller:

- `jfk.flac`
  - Source: `https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac`
  - Format: FLAC, 44.1 kHz, stereo, 11.0s.
  - SHA-256: `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`

- `micro-machines.wav`
  - Source: `https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav`
  - Format: WAV PCM s16le, 44.1 kHz, stereo, 29.888390s.
  - SHA-256: `37de21902b32aa2fc147ccbfdcc0566cc7061fffb2c0b10874f05147c0b9de0f`

Regenerate the committed WAV fixtures by downloading the source files and running:

```bash
ffmpeg -y -i jfk.flac -ac 1 -ar 16000 jfk_16khz_mono.wav
ffmpeg -y -i micro-machines.wav -ac 1 -ar 16000 micro_machines_16khz_mono.wav
```
