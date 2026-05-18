# Puppygrad TODO

## ResNet Vision Runtime Plan

Goal: implement a native ResNet image classifier first, while extracting reusable vision and CNN pieces that can later support CLIP, ViT, YOLO, OCR, and diffusion image tooling.

### Phase 1: Reusable Vision Module

- [ ] Add a shared vision/image module outside `src/models/resnet/`, for example `src/vision/`.
- [ ] Add image loading for common RGB files using a deliberate image decoding dependency.
- [ ] Convert decoded images to normalized `f32` RGB buffers.
- [ ] Add layout helpers for HWC, CHW, and NCHW-style tensors.
- [ ] Add resize helper for shortest-side resize.
- [ ] Add center-crop helper.
- [ ] Add per-channel normalization helper with mean/std.
- [ ] Keep preprocessing reusable for ResNet, CLIP, ViT, YOLO, and future image models.

### Phase 2: ResNet Model Boundary

- [ ] Add `src/models/resnet/` with `mod.rs`, `config.rs`, `weights.rs`, `model.rs`, `rust.rs`, `runtime.rs`, and any small helper files needed.
- [ ] Keep architecture/config/data structs in `model.rs`; put the Rust CPU backend implementation and kernels in `rust.rs`.
- [ ] Add `ResNetVariant` with `resnet18` as the first supported variant.
- [ ] Add `ResNetConfig` for architecture settings:
  - input channels
  - number of classes
  - stem kernel/stride/padding
  - block type
  - stage block counts
  - stage channels
  - stage strides
  - BatchNorm epsilon
  - preprocessing resize/crop/mean/std
- [ ] Hardcode the first target as ImageNet ResNet-18.
- [ ] Keep the config shape broad enough for ResNet-34/50 later without implementing those variants immediately.

### Phase 3: CNN Kernels

- [ ] Add a reference `conv2d` kernel for NCHW input and OIHW weights.
- [ ] Put the first ResNet CPU implementation in `src/models/resnet/rust.rs`.
- [ ] Support stride and padding.
- [ ] Support `1x1`, `3x3`, and `7x7` convolutions.
- [ ] Add ReLU.
- [ ] Add max-pool 2D.
- [ ] Add global average pool.
- [ ] Add residual elementwise add.
- [ ] Add linear classifier op or reuse an existing dense kernel.
- [ ] Add top-k helper for classification output.
- [ ] Keep these kernels in shared CPU/vision code when they are not ResNet-specific.

### Phase 4: BatchNorm Folding

- [ ] Load Conv + BatchNorm state and fold BatchNorm into Conv at weight-load time.
- [ ] Implement the inference folding formula:
  - `scale = gamma / sqrt(running_var + eps)`
  - `folded_weight[out] = conv_weight[out] * scale`
  - `folded_bias[out] = beta + (conv_bias[out] - running_mean) * scale`
- [ ] Support conv layers with no original bias.
- [ ] Fold downsample projection BatchNorm too.
- [ ] Store runtime weights as only folded conv weights/biases plus final FC weights/biases.
- [ ] Add unit tests for BatchNorm folding against small known tensors.

### Phase 5: Weight Loading And Assets

- [ ] Choose the first weight format path: prefer Hugging Face safetensors if available, otherwise add a documented conversion path from PyTorch weights to safetensors.
- [ ] Add asset preparation for config/weights/labels under a default directory such as `models/resnet18`.
- [ ] Load PyTorch-style keys:
  - `conv1.weight`
  - `bn1.*`
  - `layerN.B.convM.weight`
  - `layerN.B.bnM.*`
  - `layerN.B.downsample.0.weight`
  - `layerN.B.downsample.1.*`
  - `fc.weight`
  - `fc.bias`
- [ ] Validate every tensor shape against `ResNetConfig`.
- [ ] Load ImageNet class labels.
- [ ] Add clear errors for missing/extra/mis-shaped tensors.

### Phase 6: ResNet-18 Forward Pass

- [ ] Implement the stem:
  - folded `conv1`
  - ReLU
  - max pool
- [ ] Implement ResNet basic block:
  - conv3x3
  - ReLU
  - conv3x3
  - optional downsample skip
  - residual add
  - ReLU
- [ ] Implement stages:
  - layer1: 2 blocks, 64 channels
  - layer2: 2 blocks, 128 channels, first block stride 2
  - layer3: 2 blocks, 256 channels, first block stride 2
  - layer4: 2 blocks, 512 channels, first block stride 2
- [ ] Implement global average pool.
- [ ] Implement final FC classifier.
- [ ] Return logits as `[1000]`.

### Phase 7: CLI

- [ ] Add `puppygrad resnet`.
- [ ] Add `--image path`.
- [ ] Add `--variant resnet18`, defaulting to ResNet-18.
- [ ] Add `--model-dir`, defaulting to `models/resnet18`.
- [ ] Add `--download` if assets can be fetched directly.
- [ ] Add `--labels path` override.
- [ ] Add `--top-k N`, defaulting to 5.
- [ ] Add `--threads N` after the reference path works.
- [ ] Print label, probability/logit, and class index.

### Phase 8: Correctness Checks

- [ ] Add a small synthetic convolution test.
- [ ] Add pooling tests.
- [ ] Add image preprocessing tests for output shape and normalization.
- [ ] Add ResNet shape tests after each major stage.
- [ ] Compare folded Conv+BN output against unfused Conv+BN on a tiny tensor.
- [ ] Compare final logits or top-k results against PyTorch/torchvision for one fixture image.
- [ ] Add at least one small image fixture with documented source/provenance.

### Phase 9: Performance Work

- [ ] Start with a straightforward CPU reference implementation.
- [ ] Add profiling buckets:
  - image preprocessing
  - conv stem
  - layer1
  - layer2
  - layer3
  - layer4
  - global pool
  - classifier
- [ ] Parallelize output channels or spatial tiles for large convolutions.
- [ ] Reuse scratch buffers to avoid excessive allocation.
- [ ] Add a simple runtime tuning config only after correctness is stable.
- [ ] Keep GPU/backend hooks out of the first pass unless the CPU path is already correct.

### Phase 10: Reuse For Future Vision Models

- [ ] Keep image loading/preprocessing separate from ResNet-specific code.
- [ ] Keep Conv/ReLU/pool/add kernels reusable for YOLO and CNN backbones.
- [ ] Keep top-k/label output reusable for ViT classifiers.
- [ ] Note which pieces CLIP/ViT will reuse: image loading, resize/crop/normalize, labels/top-k patterns.
- [ ] Note which pieces YOLO will reuse: image loading, resize/letterbox later, conv, activations, BatchNorm folding, postprocessing foundation.
- [ ] Avoid over-generalizing transformer or detection abstractions until a second vision model needs them.

### Phase 11: Documentation

- [ ] Add README section for ResNet.
- [ ] Document supported variant and expected assets.
- [ ] Document the basic run command:
  - `puppygrad resnet --image image.jpg --top-k 5`
- [ ] Document preprocessing semantics: resize, center crop, ImageNet mean/std, RGB.
- [ ] Document known limitations: CPU reference path, ResNet-18 only, no detection yet.
- [ ] Document which reusable vision pieces are now available for future CLIP/YOLO work.

## Completion Criteria

- [ ] `cargo fmt --check` passes.
- [ ] `cargo check` passes.
- [ ] Focused ResNet/vision unit tests pass.
- [ ] Image preprocessing converts an RGB image to `[3, 224, 224]` normalized CHW data.
- [ ] Conv2D, pooling, residual add, and BatchNorm folding have deterministic unit tests.
- [ ] ResNet-18 weights load with full shape validation.
- [ ] `puppygrad resnet --image tests/data/images/example.jpg --top-k 5` prints five ImageNet classes.
- [ ] ResNet-18 top-k output for at least one fixture image matches a trusted torchvision reference closely enough for a CPU f32 implementation.
- [ ] Existing GPT-2 and Whisper commands still compile and run their smoke paths.
- [ ] README documents ResNet usage and current limitations.
