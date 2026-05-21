# Piper First Native Clip

Local voice:

- `models/piper/model.onnx`
- `models/piper/model.onnx.json`
- `en_US-lessac-medium`

Generate a native VITS WAV from raw phoneme ids:

```sh
cargo run -q -- piper \
  --model-dir models/piper \
  --phoneme-ids 1,20,59,24,27,0,35,62,24,17,2 \
  --out /tmp/puppygrad-real-hello.wav \
  --seed 1
```

The ids correspond to `^ h ə l o _ w ɜ l d $` in the local voice config.

Inspect the WAV:

```sh
cargo run -q -- audio inspect /tmp/puppygrad-real-hello.wav
```

Observed native output:

- format: PCM WAV 16-bit
- sample rate: 22050
- channels: 1
- samples: 4352
- duration: 0.197s

ONNX Runtime comparison for the same ids and default scales (`noise_scale=0.667`, `length_scale=1.0`, `noise_w=0.8`):

- output shape: `[1, 1, 1, 11776]`
- samples: 11776
- duration: 0.534s
- finite: true

The native path now runs the loaded VITS components instead of the old debug tone. Duration parity with ONNX Runtime is not exact yet.

You can also synthesize from text:

```sh
cargo run -q -- piper \
  --model-dir models/piper \
  --text "hello world" \
  --out /tmp/puppygrad-text.wav \
  --seed 1
```

For now, `--text` uses macOS `say` plus `afconvert` to write real spoken PCM WAV output. Native Piper text inference still needs the eSpeak phonemizer and tighter VITS parity; use `--phoneme-ids` to exercise the native VITS path directly.
