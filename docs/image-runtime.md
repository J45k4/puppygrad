# Image programs and Stable Diffusion 1

The generic image host accepts a prompt and copies a completed RGB8 image from a
provider. The built-in provider runs Stable Diffusion 1.x through
[`examples/image.pup`](../examples/image.pup), compiled to self-contained C.

```sh
cargo build --release
./target/release/puppygrad image examples/image.pup \
  --download --cpu-target native --threads 8 \
  --prompt 'a photograph of a golden retriever puppy in a sunny garden' \
  --width 256 --height 256 --steps 20 --seed 42 \
  --output puppy.png
```

`--download` fetches missing public SD1.5 assets from
`stable-diffusion-v1-5/stable-diffusion-v1-5`, revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, into the ignored
`models/stable-diffusion-v1-5/` directory. The fp16 weights occupy about 2.1 GB on
disk; the loader converts them to float32. Existing files are retained. Use
`--model-dir` for another compatible unsharded Diffusers safetensors checkpoint.
The default size is 512×512, default guidance is 7.5, and default steps are 20.
The model uses deterministic DDIM, even when its distributed scheduler config
specifies PNDM. `--negative-prompt` controls the unconditional text conditioning.

The current device is `cpu`. `--cpu-target generic` is the portable default;
`native` allows the host's vector instructions. Generated C, shared libraries,
and compiler manifests are cached under the ignored `.cache/pup/cpu/` directory.
The command reports each stage's kernel count, arena size, C source path, and
execution time. Arena sizes exclude weights, caller buffers, bounded GEMM weight
panels, and worker stacks. Profiling reports the panel capacity as
`packed_workspace_bytes`.

## What executes where

The `.pup` source defines CLIP attention, residual blocks, spatial transformers,
GEGLU, VAE blocks, the DDIM update, and RGB conversion. The standard library
provides convolution, group normalization, SiLU, cross-attention, and nearest
upsampling, composed from Pops. Convolution uses a `window` view and the existing
multiply/reduce matmul lowering; it does not allocate an im2col patch matrix.

The checkpoint binder specializes a static stage call graph using tensor names
and configuration dimensions. CLIP, U-Net, VAE, and DDIM compile separately and
are reused throughout a request. The host loads weights, tokenizes text,
generates initial seeded noise, selects scheduler coefficients, and sequences
the C calls. All neural-network tensor math and denoising updates execute in the
compiled C stages. Python and external inference frameworks are not used by the
image command.

`image.pup` contains stage functions. It is specialized by the image provider;
standalone `check`/`emit` currently require a complete top-level graph and bound
checkpoint context. Inspect the emitted C paths printed by `image`.

Supported: SD1 text-to-image, batch 1, CLIP's 77-token context, epsilon prediction,
leading DDIM timesteps, and packed RGB8 output. Dimensions must be divisible by
64 for the SD1.5 checkpoint. The host caps output at one megapixel; the compiler's
2 GiB workspace limit can reject large attention graphs before that cap. SDXL,
ControlNet, LoRA, inpainting, image-to-image, and GPU execution are not implemented
in this provider. Forward convolution is supported; its backward pass still
requires overlapping scatter-add support.

## FFI contract

[`include/puppygrad_image.h`](../include/puppygrad_image.h) defines ABI v1:

```c
const PupImageApi *get_image_api(void);
// api->build_model(config_json, length, error) -> opaque state
// api->infer(state, request_json, length, callbacks, error) -> status
// api->read_output(state, NULL, 0, &info, error) -> metadata query
// api->read_output(state, caller_buffer, capacity, &info, error) -> copy
// api->free_model(state)
```

A shared library can use the same host:

```sh
./target/release/puppygrad image ./model.so --prompt 'a puppy' --output puppy.png
```

Configuration JSON contains `source`, `model_dir`, `device`, `threads`, and
`cpu_target`. Request JSON contains `prompt`, `negative_prompt`, `width`, `height`,
`steps`, `guidance_scale`, and `seed`. These fields are the built-in provider's
request schema; other providers must implement the same schema to work with this
CLI. `--download` and nondefault `--cpu-target` apply to `.pup` providers.

Calls on one state are exclusive and synchronous. Providers finish callbacks
before `infer` returns. Optional `on_progress(user, step, total)` reports monotonic
steps; optional `on_done(user, status)` fires exactly once and matches the return
status. Step progress counts denoising steps; image decoding follows the last
step, and completion follows decoding. Status zero means success.

`read_output(NULL, 0, ...)` queries dimensions, format, stride, and required bytes.
The host allocates its own buffer and calls again to copy pixels. Format 1 means
packed RGB8 with `stride = width * 3`. The provider retains output until the next
`infer` or `free_model`; an unsuccessful request leaves no completed image.
The caller frees its buffer independently. `free_model` releases provider state
before the library is unloaded. Callbacks must not reenter the same state.

## Validation

```sh
cargo test --test compiler_image --test image_ffi
# Optional downloaded-checkpoint parity; no network activity inside tests:
cargo test --test image_stages -- --ignored --nocapture
```

The optional test expects the public
`hf-internal-testing/tiny-stable-diffusion-pipe` weights converted to canonical
safetensors paths under `models/stable-diffusion-tiny/`. This checkpoint contains
random small weights and tests numerical execution rather than image quality.
It compares compiled CLIP, U-Net at two timesteps, and VAE outputs against the
existing native numerical reference. Normal tests independently check strided
convolution, padding, view coordinates, normalization, sine gradients, buffer
ownership, callbacks, malformed metadata, and a real C shared-library provider.
