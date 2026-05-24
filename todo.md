# Puppygrad TODO

## Goal: Stable Diffusion Model Support

Add text-to-image Stable Diffusion support to Puppygrad. The implementation should give users a working generation path early through a Python Diffusers backend, then grow into a native Rust reference runtime that can run Stable Diffusion 1.x without calling Python, Torch, Diffusers, Transformers, ONNX Runtime, or another model execution process.

First target: Stable Diffusion 1.x / 1.5 in Hugging Face Diffusers directory layout, batch size 1, text-to-image, 512 x 512, classifier-free guidance, and a DDIM scheduler.

Out of initial scope until the SD 1.x text-to-image path works end-to-end:

- SDXL
- img2img
- inpainting
- ControlNet
- LoRA
- textual inversion
- safety checker execution
- batched generation
- GPU acceleration
- quantized/native low-memory execution

Primary correctness objective:

- Provide a `stable-diffusion` command that generates an image from a prompt and saves PNG or JPEG output.
- Keep the Python Diffusers backend as an external backend and parity reference, not as a dependency of native Rust inference.
- Build native Rust inference correctness-first, even if the first CPU path is slow.
- Validate native parity using tensor-level fixtures and tolerances rather than requiring byte-identical encoded image files.
- Compare prompt token ids, text embeddings, scheduler timesteps, latent shapes, selected UNet noise predictions, final latent statistics, decoded image statistics, and deterministic tensor slices against Python Diffusers references.
- Require exact shape/layout parity for all model inputs and outputs.
- Require deterministic native output for the same backend, model files, seed, prompt, negative prompt, scheduler, step count, guidance scale, width, and height on the same machine.
- Document all differences between native Rust and Python Diffusers when exact floating-point parity is not realistic.

Target user command for the first working backend:

- `./target/release/puppygrad stable-diffusion --backend python-diffusers --download --prompt "a corgi in a spacesuit" --out /tmp/corgi.png --steps 25 --seed 42`

Target native command after Rust runtime is implemented:

- `./target/release/puppygrad stable-diffusion --backend rust --model-dir models/stable-diffusion-v1-5 --prompt "a corgi in a spacesuit" --negative-prompt "blurry, low quality" --out /tmp/corgi-rust.png --steps 25 --guidance-scale 7.5 --seed 42`

## 1. Define Stable Diffusion Scope And Runtime Contract

- [ ] Select and document the first supported model family: Stable Diffusion 1.x / 1.5 Diffusers-format checkpoints.
- [ ] Select the default model id for the user-facing command.
  - [ ] Evaluate whether `runwayml/stable-diffusion-v1-5` is acceptable given license and access requirements.
  - [ ] Identify a smaller permissive test checkpoint for local fixtures and CI-like smoke tests.
  - [ ] Avoid assuming gated Hugging Face access is available.
- [ ] Document that users are responsible for complying with model licenses and access terms.
- [ ] Define `StableDiffusionRuntimeOptions` with:
  - [ ] prompt
  - [ ] optional negative prompt
  - [ ] output image path
  - [ ] width
  - [ ] height
  - [ ] step count
  - [ ] guidance scale
  - [ ] seed
  - [ ] scheduler choice
  - [ ] backend choice
  - [ ] optional model directory
  - [ ] model id
  - [ ] revision
  - [ ] download flag
  - [ ] output format or format inferred from file extension
  - [ ] stats/progress flag
- [ ] Define supported dimension rules.
  - [ ] Default to 512 x 512.
  - [ ] Require width and height to be multiples of 8.
  - [ ] Reject zero dimensions.
  - [ ] Reject dimensions that are too large for the initial reference implementation unless an explicit override is added later.
- [ ] Define backend behavior.
  - [ ] `python-diffusers` may invoke Python and use Torch/Diffusers.
  - [ ] `rust` must run native Rust model code only.
  - [ ] Future GPU support must be represented as an explicit backend, not hidden inside `rust`.
- [ ] Define determinism expectations.
  - [ ] Same backend and same options should be reproducible on the same machine/dependency versions.
  - [ ] Cross-backend bit identity is not required.
  - [ ] Native parity should be tensor/tolerance based.
- [ ] Add clear errors for unsupported models, unsupported configs, missing assets, invalid dimensions, empty prompts, unavailable backends, and missing Python packages.
- [ ] Keep CLI-only file handling and user-facing messages in `src/main.rs`.
- [ ] Keep Stable Diffusion runtime code under `src/models/stable_diffusion/`.

## 2. Wire The CLI

- [ ] Add `StableDiffusion` to the top-level `Command` enum in `src/main.rs`.
- [ ] Add `StableDiffusionBackendArg` with:
  - [ ] `PythonDiffusers`
  - [ ] `Rust`
- [ ] Add `StableDiffusionSchedulerArg` with:
  - [ ] `Ddim` as the first native target
  - [ ] `Euler` as a later usability target
  - [ ] `Ddpm` only if useful for tests or fixture parity
- [ ] Add CLI args:
  - [ ] `--model-dir`
  - [ ] `--model-id`
  - [ ] `--revision`
  - [ ] `--download`
  - [ ] `--backend`
  - [ ] `--prompt`
  - [ ] `--negative-prompt`
  - [ ] `--out`
  - [ ] `--steps`
  - [ ] `--guidance-scale`
  - [ ] `--width`
  - [ ] `--height`
  - [ ] `--seed`
  - [ ] `--scheduler`
  - [ ] `--stats`
- [ ] Validate prompt is non-empty after trimming whitespace.
- [ ] Default negative prompt to an empty string.
- [ ] Validate `steps > 0`.
- [ ] Validate `guidance_scale >= 0`.
- [ ] Validate width and height are positive multiples of 8.
- [ ] Infer image format from `--out` extension.
- [ ] Support at least `.png`, `.jpg`, and `.jpeg`.
- [ ] Prefer PNG in examples and docs.
- [ ] Print concise status to stderr after successful generation:
  - [ ] backend
  - [ ] model directory or model id
  - [ ] seed
  - [ ] dimensions
  - [ ] scheduler
  - [ ] steps
  - [ ] guidance scale
  - [ ] output path
  - [ ] elapsed time when available
- [ ] Ensure binary image bytes are never written to stdout.
- [ ] Add CLI help text that distinguishes external and native backends.

## 3. Add Stable Diffusion Module Layout

- [ ] Add `src/models/stable_diffusion/mod.rs`.
- [ ] Add `pub mod stable_diffusion;` to `src/models/mod.rs`.
- [ ] Add `src/models/stable_diffusion/error.rs` for typed errors.
- [ ] Add `src/models/stable_diffusion/assets.rs` for model ids, required files, optional files, and asset preparation.
- [ ] Add `src/models/stable_diffusion/config.rs` for typed JSON config structs.
- [ ] Add `src/models/stable_diffusion/weights.rs` for safetensors loading and tensor namespace mapping.
- [ ] Add `src/models/stable_diffusion/runtime.rs` for backend dispatch and public runtime entry points.
- [ ] Add `src/models/stable_diffusion/python.rs` for the Python Diffusers backend.
- [ ] Add `src/models/stable_diffusion/pipeline.rs` for native end-to-end orchestration.
- [ ] Add `src/models/stable_diffusion/scheduler.rs` for DDIM and future schedulers.
- [ ] Add `src/models/stable_diffusion/clip.rs` for CLIP text encoder inference.
- [ ] Add `src/models/stable_diffusion/unet.rs` for UNet2DConditionModel inference.
- [ ] Add `src/models/stable_diffusion/vae.rs` for AutoencoderKL decoder inference.
- [ ] Add `src/models/stable_diffusion/tensor.rs` if existing tensor helpers are not general enough.
- [ ] Keep reusable low-level operations in shared modules when they are not Stable-Diffusion-specific.
- [ ] Avoid routing inference through the autograd `engine`; use direct inference kernels.

## 4. Prepare Stable Diffusion Assets

- [ ] Define expected Diffusers model directory layout:
  - [ ] `model_index.json`
  - [ ] `scheduler/scheduler_config.json`
  - [ ] `tokenizer/tokenizer.json` when available
  - [ ] `tokenizer/vocab.json`
  - [ ] `tokenizer/merges.txt`
  - [ ] `tokenizer/tokenizer_config.json`
  - [ ] `text_encoder/config.json`
  - [ ] `text_encoder/model.safetensors`
  - [ ] `unet/config.json`
  - [ ] `unet/diffusion_pytorch_model.safetensors`
  - [ ] `vae/config.json`
  - [ ] `vae/diffusion_pytorch_model.safetensors`
- [ ] Confirm exact filenames for the selected default model.
- [ ] Decide how to handle `.bin` weights.
  - [ ] Prefer safetensors for native Rust.
  - [ ] Allow Python Diffusers to use whatever Diffusers supports.
  - [ ] Report a clear native-backend error if only unsupported `.bin` weights are available.
- [ ] Decide how to handle sharded safetensors.
  - [ ] Prefer unsharded files for the first native implementation.
  - [ ] Add safetensors index JSON support if the selected default model requires shards.
- [ ] Extend asset validation to handle nested required files cleanly.
- [ ] Reuse `download_huggingface_file` for nested model files.
- [ ] Add optional asset support for files useful to Diffusers but not required by native inference.
- [ ] Add `default_stable_diffusion_dir()` returning a path under `models/`.
- [ ] Add `prepare_stable_diffusion_assets()` analogous to existing model asset preparation functions.
- [ ] Support local-only use when assets are present and `--download` is omitted.
- [ ] Add clear errors for failed downloads, missing revisions, private/gated models, and network issues.
- [ ] Add a debug/manifest path that can list found tensor names, dtypes, shapes, and storage sizes.
- [ ] Ensure no model weights or generated images are committed to git unless they are tiny test fixtures explicitly intended for source control.

## 5. Implement Python Diffusers Backend MVP

- [ ] Implement `python-diffusers` as the first working end-to-end Stable Diffusion backend.
- [ ] Follow the existing Bark external-backend pattern.
- [ ] Keep Python-specific logic contained in `src/models/stable_diffusion/python.rs` and CLI dispatch code.
- [ ] Pass arguments to Python using explicit subprocess arguments, not shell interpolation.
- [ ] Support passing:
  - [ ] model directory or model id
  - [ ] revision
  - [ ] download/cache preference
  - [ ] prompt
  - [ ] negative prompt
  - [ ] output path
  - [ ] width
  - [ ] height
  - [ ] steps
  - [ ] guidance scale
  - [ ] seed
  - [ ] scheduler
- [ ] Load a local pipeline from `--model-dir` when provided.
- [ ] Use Hugging Face/Diffusers model loading only when `--download` or model id based execution is requested.
- [ ] Configure the requested scheduler in Python.
- [ ] Set the Torch generator seed deterministically.
- [ ] Save the generated image to exactly the path requested by Rust.
- [ ] Return or write small metadata for Rust to verify:
  - [ ] backend
  - [ ] model source
  - [ ] width
  - [ ] height
  - [ ] seed
  - [ ] steps
  - [ ] scheduler
  - [ ] guidance scale
  - [ ] elapsed time
- [ ] Validate that the output file exists and is non-empty after Python exits.
- [ ] Surface Python stderr clearly on failure.
- [ ] Detect missing Python packages and print a suggested install command.
- [ ] Do not automatically install packages.
- [ ] Document required Python packages:
  - [ ] `diffusers`
  - [ ] `torch`
  - [ ] `transformers`
  - [ ] `safetensors`
  - [ ] `Pillow`
- [ ] Add a local smoke command for the Python backend.
- [ ] Add README documentation for the Python backend and its limitations.

## 6. Add Native Tensor And Math Primitives

- [ ] Decide whether to reuse `vision::cnn::NchwTensor` or introduce a more general inference tensor type.
- [ ] Add or extend tensor representation for:
  - [ ] 1D vectors
  - [ ] 2D matrices
  - [ ] 3D sequence tensors
  - [ ] 4D NCHW tensors
- [ ] Add runtime shape validation helpers that return clear errors instead of panicking.
- [ ] Add f16 and bf16 safetensors conversion support.
  - [ ] Add the `half` crate if appropriate.
  - [ ] Convert `F16` and `BF16` to f32 on load for the first native path.
- [ ] Add tensor finite checks.
- [ ] Add tensor statistics helpers:
  - [ ] min
  - [ ] max
  - [ ] mean
  - [ ] standard deviation
  - [ ] RMS
  - [ ] selected slice extraction
- [ ] Add elementwise operations:
  - [ ] add
  - [ ] sub
  - [ ] mul
  - [ ] div where needed
  - [ ] scalar multiply
  - [ ] scalar add
  - [ ] clamp
- [ ] Add broadcasting patterns needed by SD configs.
- [ ] Add channel-wise affine transforms.
- [ ] Add concatenation along channel and sequence dimensions.
- [ ] Add split/chunk helpers for attention projections and classifier-free guidance batches.
- [ ] Add NCHW nearest-neighbor upsample by integer factor.
- [ ] Add downsample helpers.
- [ ] Improve or add conv2d support for UNet and VAE:
  - [ ] 1 x 1 kernels
  - [ ] 3 x 3 kernels
  - [ ] arbitrary input/output channels
  - [ ] stride 1
  - [ ] stride 2
  - [ ] symmetric padding
  - [ ] optional bias
- [ ] Add GroupNorm for NCHW tensors.
- [ ] Add LayerNorm for sequence tensors if existing helpers are not flexible enough.
- [ ] Add SiLU activation.
- [ ] Add GELU variant required by CLIP.
- [ ] Add batched matrix multiplication.
- [ ] Add multi-head self-attention over sequence tensors.
- [ ] Add cross-attention where query length differs from key/value length.
- [ ] Add numerically stable softmax for attention.
- [ ] Add sinusoidal timestep embedding helpers.
- [ ] Add deterministic normal random latent initialization from a seed.
- [ ] Add unit tests for each primitive using small handcrafted inputs.
- [ ] Add Python-generated reference fixtures for GroupNorm, attention, scheduler math, and f16 conversion.

## 7. Load And Validate Native Configs

- [ ] Define typed config structs for `model_index.json`.
- [ ] Define typed config structs for `text_encoder/config.json`.
- [ ] Define typed config structs for `unet/config.json`.
- [ ] Define typed config structs for `vae/config.json`.
- [ ] Define typed config structs for `scheduler/scheduler_config.json`.
- [ ] Ignore unknown fields safely, but preserve enough metadata for debug output.
- [ ] Validate architecture class names match supported SD 1.x components.
- [ ] Validate CLIP config fields:
  - [ ] vocab size
  - [ ] hidden size
  - [ ] intermediate size
  - [ ] layer count
  - [ ] attention head count
  - [ ] max position embeddings
  - [ ] activation
  - [ ] layer norm epsilon
- [ ] Validate UNet config fields:
  - [ ] sample size
  - [ ] input channels
  - [ ] output channels
  - [ ] block out channels
  - [ ] down block types
  - [ ] up block types
  - [ ] layers per block
  - [ ] cross-attention dimension
  - [ ] attention head dimensions/counts
  - [ ] normalization groups
- [ ] Validate VAE config fields:
  - [ ] latent channels
  - [ ] scaling factor
  - [ ] block channels
  - [ ] up/down block layout
  - [ ] normalization groups
- [ ] Validate scheduler config fields:
  - [ ] beta schedule
  - [ ] beta start/end
  - [ ] training timestep count
  - [ ] clip sample behavior
  - [ ] prediction type
  - [ ] timestep spacing
- [ ] Add clear unsupported-config errors for SDXL, inpainting UNets, ControlNet, non-CLIP text encoders, unsupported schedulers, and unexpected channel layouts.
- [ ] Add unit tests for supported and unsupported config loading.

## 8. Load Native Weights

- [ ] Extend safetensors helpers to read f32, f16, and bf16 tensors as f32 vectors.
- [ ] Add typed CLIP text encoder weight structs.
- [ ] Add typed UNet weight structs.
- [ ] Add typed VAE decoder weight structs.
- [ ] Map Diffusers tensor names into typed structs.
- [ ] Validate every required tensor:
  - [ ] present
  - [ ] supported dtype
  - [ ] expected rank
  - [ ] expected shape
- [ ] Add clear errors for missing tensors, wrong dtype, wrong rank, wrong shape, and unsupported checkpoint layout.
- [ ] Support optional biases where config/checkpoint variants allow them.
- [ ] Add a native tensor manifest/debug mode for inspecting names and shapes.
- [ ] Add handcrafted tiny safetensors unit tests for successful and failing loads.
- [ ] Add local ignored real-checkpoint load tests gated on model files existing.

## 9. Implement Tokenization And CLIP Text Encoder

- [ ] Determine whether the existing `tokenizers` crate path can load the selected tokenizer assets directly.
- [ ] Support prompt and negative prompt tokenization with max length 77 for SD 1.x.
- [ ] Match Diffusers/Transformers padding, truncation, BOS/EOS, and unknown-token behavior.
- [ ] Add tokenizer fixtures for:
  - [ ] empty negative prompt
  - [ ] short prompt
  - [ ] punctuation
  - [ ] long prompt requiring truncation
  - [ ] mixed case
  - [ ] non-ASCII text
- [ ] Implement CLIP token embedding lookup.
- [ ] Implement CLIP positional embeddings.
- [ ] Implement CLIP transformer encoder blocks:
  - [ ] pre/post layer norm order as configured
  - [ ] self-attention
  - [ ] MLP
  - [ ] residual connections
  - [ ] causal or attention mask behavior as required by CLIP text model
- [ ] Return text embeddings in the shape expected by UNet cross-attention.
- [ ] Run both conditional and unconditional prompt embeddings for classifier-free guidance.
- [ ] Add shape tests for token ids and embeddings.
- [ ] Add parity tests against Python Diffusers/Transformers for token ids and selected text embedding slices.

## 10. Implement Scheduler Support

- [ ] Implement DDIM scheduler first.
- [ ] Load scheduler config from `scheduler/scheduler_config.json`.
- [ ] Build beta, alpha, cumulative alpha, and sigma/timestep arrays.
- [ ] Implement timestep selection for inference steps.
- [ ] Implement latent scaling before UNet input when required by scheduler.
- [ ] Implement DDIM step update from predicted noise to previous latent.
- [ ] Support prediction type variants required by SD 1.x.
- [ ] Validate scheduler math against small Python fixtures.
- [ ] Add tests for timestep counts and monotonic ordering.
- [ ] Add tests for one scheduler step with known tiny tensors.
- [ ] Defer Euler scheduler until DDIM works end-to-end.

## 11. Implement UNet2DConditionModel

- [ ] Implement timestep projection and timestep embedding MLP.
- [ ] Implement ResNet blocks used by SD 1.x UNet.
- [ ] Implement down blocks:
  - [ ] plain down block
  - [ ] cross-attention down block
  - [ ] downsample operation
  - [ ] skip/residual capture
- [ ] Implement middle block:
  - [ ] ResNet block
  - [ ] self-attention/cross-attention as required
  - [ ] second ResNet block
- [ ] Implement up blocks:
  - [ ] skip tensor concatenation
  - [ ] plain up block
  - [ ] cross-attention up block
  - [ ] upsample operation
- [ ] Implement transformer blocks inside attention modules:
  - [ ] group norm or layer norm as required
  - [ ] self-attention
  - [ ] cross-attention over text embeddings
  - [ ] feed-forward network
  - [ ] residual connections
- [ ] Implement final normalization, activation, and output convolution.
- [ ] Ensure UNet input and output latent shapes match.
- [ ] Support classifier-free guidance batch size 2 internally or run unconditional/conditional passes separately first for simplicity.
- [ ] Add shape trace/debug output for every major UNet stage.
- [ ] Add tiny synthetic UNet tests where possible.
- [ ] Add local real-checkpoint parity tests comparing selected noise prediction slices to Python Diffusers.

## 12. Implement VAE Decoder

- [ ] Load VAE decoder config and weights.
- [ ] Apply Stable Diffusion latent scaling factor before decode, typically dividing by `0.18215` when appropriate.
- [ ] Implement VAE post-quant convolution if present.
- [ ] Implement decoder ResNet blocks.
- [ ] Implement decoder upsampling blocks.
- [ ] Implement VAE attention block if present in the selected architecture.
- [ ] Implement final normalization, activation, and output convolution.
- [ ] Convert decoded tensor range to RGB image range following Diffusers behavior.
- [ ] Clamp decoded image values consistently with reference behavior.
- [ ] Validate output tensor shape `[1, 3, height, width]`.
- [ ] Add synthetic VAE decoder shape tests.
- [ ] Add local real-checkpoint parity tests comparing selected decoded image slices and image statistics.

## 13. Implement Native Pipeline Orchestration

- [ ] Implement native `StableDiffusionPipeline` construction from a model directory.
- [ ] Load configs, tokenizer, CLIP weights, UNet weights, VAE weights, and scheduler.
- [ ] Tokenize prompt and negative prompt.
- [ ] Encode prompt and unconditional embeddings.
- [ ] Initialize latents with deterministic seeded normal noise shaped `[1, 4, height / 8, width / 8]`.
- [ ] Apply scheduler initial noise sigma if required.
- [ ] For each denoising step:
  - [ ] prepare scheduler-scaled latent input
  - [ ] evaluate UNet for unconditional and conditional embeddings
  - [ ] combine noise predictions using classifier-free guidance
  - [ ] update latents with scheduler step
  - [ ] optionally emit progress/stats
- [ ] Decode final latents with VAE.
- [ ] Convert decoded tensor to an RGB image.
- [ ] Save output image.
- [ ] Return generation metadata to the caller.
- [ ] Add cancellation/progress hooks later only after the basic path works.
- [ ] Add debug dump options for parity development, kept out of normal user output.

## 14. Image Output And Metadata

- [ ] Reuse the `image` crate for PNG/JPEG output.
- [ ] Add RGB tensor to image-buffer conversion helpers.
- [ ] Validate output parent directory behavior and error messages.
- [ ] Avoid overwriting concerns unless a future `--no-clobber` flag is requested.
- [ ] Write image metadata sidecars only when explicitly requested or useful for tests.
- [ ] Add optional JSON metadata output later if needed for experiments.
- [ ] Add tests for output format detection.
- [ ] Add tests that generated image buffers have expected dimensions and finite/clamped values.

## 15. Python Reference Fixtures And Parity Tools

- [ ] Add a Python fixture generator for Stable Diffusion intermediates.
- [ ] Keep fixture generation outside the Rust runtime path.
- [ ] Make the fixture generator accept:
  - [ ] model directory or model id
  - [ ] prompt
  - [ ] negative prompt
  - [ ] seed
  - [ ] scheduler
  - [ ] steps
  - [ ] guidance scale
  - [ ] width
  - [ ] height
  - [ ] output fixture directory
- [ ] Capture fixture metadata:
  - [ ] model id or local path
  - [ ] revision when available
  - [ ] Diffusers version
  - [ ] Transformers version
  - [ ] Torch version
  - [ ] Python version
  - [ ] seed
  - [ ] prompt
  - [ ] negative prompt
  - [ ] scheduler
  - [ ] steps
  - [ ] guidance scale
  - [ ] dimensions
- [ ] Capture tokenizer reference data:
  - [ ] conditional token ids
  - [ ] unconditional token ids
  - [ ] attention masks if used
- [ ] Capture CLIP reference data:
  - [ ] conditional embedding shape
  - [ ] unconditional embedding shape
  - [ ] selected embedding slices
  - [ ] embedding statistics
- [ ] Capture scheduler reference data:
  - [ ] timesteps
  - [ ] alpha/beta arrays or selected slices
  - [ ] initial noise sigma
- [ ] Capture denoising reference data:
  - [ ] initial latent shape and selected slices
  - [ ] UNet input shape for first step
  - [ ] first-step unconditional noise prediction slice
  - [ ] first-step conditional noise prediction slice
  - [ ] first-step guided noise prediction slice
  - [ ] latent slice after first scheduler step
  - [ ] final latent statistics
- [ ] Capture VAE/image reference data:
  - [ ] decoded tensor shape
  - [ ] decoded tensor selected slices
  - [ ] decoded tensor statistics
  - [ ] final image dimensions
  - [ ] final image pixel sample slices
- [ ] Store only small fixtures in git.
- [ ] Keep large full latent/image fixtures local-only and ignored by git.
- [ ] Add Rust fixture readers for JSON and binary float slices.
- [ ] Add ignored/local end-to-end comparison script that can run both Python and Rust backends and summarize differences.

## 16. Testing Plan

- [ ] Add unit tests for CLI option validation helpers.
- [ ] Add unit tests for asset path and missing-file reporting.
- [ ] Add unit tests for config loading and unsupported-config errors.
- [ ] Add unit tests for f16/bf16 safetensors conversion.
- [ ] Add unit tests for tensor shapes and broadcasting helpers.
- [ ] Add unit tests for conv2d, upsample, downsample, GroupNorm, LayerNorm, SiLU, GELU, attention, and batched matmul.
- [ ] Add unit tests for DDIM scheduler timestep creation and one-step update.
- [ ] Add tokenizer parity tests using small checked-in fixtures.
- [ ] Add CLIP selected-slice parity tests using small checked-in fixtures.
- [ ] Add local ignored tests for real-checkpoint CLIP execution.
- [ ] Add local ignored tests for real-checkpoint UNet selected-slice parity.
- [ ] Add local ignored tests for real-checkpoint VAE selected-slice parity.
- [ ] Add local ignored native end-to-end test that writes an image.
- [ ] Add Python backend smoke test that is skipped when Python packages are unavailable.
- [ ] Keep full SD model tests skipped by default unless model assets are present.
- [ ] Ensure `cargo fmt --check` passes.
- [ ] Ensure `cargo test stable_diffusion` passes without model assets for unit tests.
- [ ] Ensure full `cargo test` passes in the default no-hardware feature set.

## 17. Performance Plan After Correctness

- [ ] Profile native CLIP execution.
- [ ] Profile native UNet execution by block.
- [ ] Profile native VAE decode.
- [ ] Profile scheduler and pipeline overhead.
- [ ] Identify avoidable allocations in the denoising loop.
- [ ] Reuse latent, attention, convolution, and activation buffers where practical.
- [ ] Add thread-pool parallelism only after single-thread correctness is validated.
- [ ] Improve conv2d performance using im2col plus existing `gemm` where appropriate.
- [ ] Improve attention performance with chunking and buffer reuse.
- [ ] Consider storing weights in f16 while accumulating in f32 after f32 correctness is stable.
- [ ] Add optional low-memory mode only after baseline native correctness works.
- [ ] Document expected runtime for:
  - [ ] Python Diffusers backend on a typical GPU/CPU environment
  - [ ] native Rust CPU reference path for a small step count
  - [ ] native Rust CPU reference path for 512 x 512 and 25 steps when feasible

## 18. Documentation

- [ ] Update `readme.md` model table with Stable Diffusion status.
- [ ] Document Python backend usage.
- [ ] Document native backend status and limitations.
- [ ] Document required Python packages for `python-diffusers`.
- [ ] Document model asset layout.
- [ ] Document how to use a local Diffusers model directory.
- [ ] Document Hugging Face access and licensing caveats.
- [ ] Document deterministic generation options.
- [ ] Document known limitations:
  - [ ] SD 1.x only initially
  - [ ] batch size 1
  - [ ] text-to-image only
  - [ ] CPU native path expected to be slow
  - [ ] no safety checker execution in native path initially
- [ ] Add troubleshooting section for:
  - [ ] missing Python packages
  - [ ] missing model files
  - [ ] gated model download failures
  - [ ] unsupported dimensions
  - [ ] unsupported checkpoint layouts
  - [ ] out-of-memory or very slow native runs

## 19. Milestone Breakdown

- [ ] Milestone A: CLI shell and module skeleton compile.
  - [ ] `stable-diffusion --help` works.
  - [ ] invalid options produce clear errors.
  - [ ] asset path helpers are unit tested.
- [ ] Milestone B: Python Diffusers backend works.
  - [ ] Generates an image from a prompt.
  - [ ] Saves PNG output.
  - [ ] Reports metadata.
  - [ ] README contains usage instructions.
- [ ] Milestone C: Native configs and weights load.
  - [ ] Loads CLIP, UNet, VAE, and scheduler configs.
  - [ ] Loads safetensors weights as typed structs.
  - [ ] Produces useful errors for unsupported layouts.
- [ ] Milestone D: Native CLIP parity.
  - [ ] Token ids match Python fixtures.
  - [ ] Text embedding slices match within tolerance.
- [ ] Milestone E: Native scheduler parity.
  - [ ] Timesteps match Python fixtures.
  - [ ] One-step DDIM updates match tiny fixtures.
- [ ] Milestone F: Native VAE decode parity.
  - [ ] Decodes fixed latent fixtures.
  - [ ] Image tensor slices/statistics match Python within tolerance.
- [ ] Milestone G: Native UNet selected-slice parity.
  - [ ] First-step UNet noise prediction slices match Python within tolerance.
- [ ] Milestone H: Native end-to-end image generation.
  - [ ] Generates finite image tensors.
  - [ ] Saves a valid image.
  - [ ] Produces stable output for a fixed seed.
- [ ] Milestone I: Native performance cleanup.
  - [ ] Removes avoidable allocations.
  - [ ] Adds targeted threading.
  - [ ] Documents expected runtime.

## Completion crieteria

- [x] `puppygrad stable-diffusion --backend python-diffusers --prompt "hello" --out /tmp/sd.png` writes a valid image when Python dependencies and model assets are available.
- [x] `puppygrad stable-diffusion --backend rust --prompt "hello" --out /tmp/sd-rust.png` writes a valid image when native-supported model assets are available.
- [x] The Rust backend does not execute Python, Torch, Diffusers, Transformers, ONNX Runtime, or another model process.
- [x] The output image has the requested dimensions and finite/clamped RGB values.
- [x] Prompt and negative prompt token ids match Python reference fixtures.
- [x] CLIP text embedding selected slices match Python reference fixtures within documented tolerances.
- [x] Scheduler timesteps and one-step updates match Python reference fixtures within documented tolerances.
- [x] UNet selected noise prediction slices match Python reference fixtures within documented tolerances.
- [x] VAE decoded tensor slices and image statistics match Python reference fixtures within documented tolerances.
- [x] Native end-to-end generation is deterministic for a fixed seed and settings on the same machine.
- [x] Missing assets produce clear errors with expected file paths.
- [x] Unsupported configs produce clear errors naming the unsupported feature.
- [x] `cargo fmt --check` passes.
- [x] Stable Diffusion unit tests pass.
- [x] Full `cargo test` passes in the default no-hardware feature set.
