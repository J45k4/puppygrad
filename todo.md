# Puppygrad TODO

## Realtime Video Capture And Vision Plan

Goal: add shared camera/video capture utilities, then wire captured frames into the existing still-image vision path. ResNet remains the first model target for frame classification, while the video layer should stay reusable for CLIP, YOLO, OCR, and future multimodal pipelines.

### Phase 1: Video Module Boundary

- [x] Add a shared video module outside `src/models/`, for example `src/video/`.
- [x] Define a `VideoError` / `VideoResult` type.
- [x] Define a shared `VideoDeviceInfo` type with index, display name, backend, and default marker if available.
- [x] Define a shared `VideoFrame` type with width, height, pixel format, timestamp, and RGB/RGBA bytes.
- [x] Keep camera/device logic out of `src/models/resnet/`.
- [x] Keep still-image preprocessing in `src/vision/` reusable for both image files and video frames.

### Phase 2: Camera Backend Choice

- [x] Choose an initial camera backend:
  - `nokhwa` for cross-platform MVP if it works well enough.
  - platform-specific AVFoundation later if macOS behavior needs tighter control.
- [x] Add the dependency deliberately and document why it was chosen.
- [x] Verify the backend can list cameras on macOS.
- [x] Verify the backend can capture RGB frames or convert frames to RGB.
- [x] Keep dependency features minimal.

### Phase 3: Video CLI Namespace

- [x] Add a top-level `video` CLI command group.
- [x] Add `puppygrad video list-devices`.
- [x] Add `puppygrad video capture-frame --out frame.jpg`.
- [x] Add `puppygrad video capture-frame --device N --out frame.jpg`.
- [x] Add `puppygrad video stream --fps N` as a basic frame counter/status smoke test.
- [x] Keep video utility commands independent from ResNet or other model assets.

### Phase 4: Frame Capture

- [x] Implement one-shot frame capture from the default camera.
- [x] Support optional camera index selection.
- [x] Support requested resolution if the backend allows it.
- [x] Support requested FPS if the backend allows it.
- [x] Convert captured frames to RGB8.
- [x] Save captured frames to PNG or JPEG through existing image tooling.
- [x] Add clear errors for camera permission denial, missing device, unsupported format, and capture timeout.

### Phase 5: Continuous Video Stream

- [x] Implement continuous frame capture until Ctrl-C.
- [x] Add a bounded frame queue independent from model inference speed.
- [x] Add queue overflow policy:
  - drop oldest
  - drop newest
  - block
- [x] Default to dropping old frames for realtime vision.
- [x] Track capture FPS, processing FPS, queue depth, and dropped frame count.
- [x] Print stream stats to stderr when requested.

### Phase 6: Vision Preprocessing From Frames

- [x] Add a conversion path from `VideoFrame` to the existing `src/vision/` RGB image type.
- [x] Reuse resize, center crop, CHW conversion, and normalization for frame classification.
- [x] Avoid writing frames to disk for model inference.
- [x] Keep file-image and camera-frame preprocessing numerically consistent.
- [x] Add tests for frame-to-RGB and RGB-to-CHW conversion without requiring a camera.

### Phase 7: ResNet On Camera Frames

- [x] Add a command for classifying a captured frame:
  - `puppygrad resnet --camera`
- [x] Add optional camera selection:
  - `puppygrad resnet --camera --device N`
- [x] Make `--image` and `--camera` mutually exclusive.
- [x] Add continuous frame classification:
  - `puppygrad resnet --camera --stream --fps 1`
- [x] Reuse the existing ResNet runtime and preprocessing config.
- [x] Print top-k labels per processed frame.
- [x] Keep ResNet output clearly described as whole-frame classification, not object detection.

### Phase 8: Machine-Readable Video Events

- [x] Add an event output format for video/model streams, likely newline-delimited JSON.
- [x] Emit frame classification events with timestamp, frame index, labels, scores, and processing latency.
- [x] Keep status/warnings on stderr.
- [x] Make stdout suitable for piping into future agents or logging tools.
- [x] Preserve human-readable output as the default.

### Phase 9: Performance And Latency

- [x] Start with low FPS defaults such as 1 FPS for ResNet CPU classification.
- [x] Add `--fps N` to control capture/processing rate.
- [x] Add `--max-queued-frames N`, default small such as 2.
- [x] Add `--drop-policy oldest|newest|block`, default `oldest`.
- [ ] Reuse ResNet preprocessing buffers where practical.
- [x] Add per-stage timing for capture, preprocessing, inference, and output.
- [x] Warn when model inference falls behind requested FPS.

### Phase 10: Future Vision Models

- [x] Keep the video module model-agnostic so CLIP can classify/score frames later.
- [x] Keep the video module compatible with YOLO frame detection later.
- [ ] Add a future `puppygrad yolo --camera --stream` path after detection exists.
- [ ] Add optional frame sampling policy for expensive models.
- [x] Add optional recording/snapshot utilities if needed.
- [x] Avoid adding detection-specific APIs until YOLO or another detector is implemented.

### Phase 11: Tests And Manual Verification

- [x] Unit test frame conversion without requiring a real camera.
- [x] Unit test queue overflow policies.
- [x] Keep actual camera tests as manual smoke tests.
- [x] Verify `puppygrad video list-devices` on macOS.
- [x] Verify `puppygrad video capture-frame --out /tmp/puppygrad-frame.jpg`.
- [x] Verify `puppygrad video stream --fps 1` runs until Ctrl-C.
- [x] Verify `puppygrad resnet --camera --stream --fps 1 --top-k 3` classifies frames if ResNet assets are present.
- [x] Verify existing image-file ResNet path still works.

### Phase 12: Documentation

- [x] Document `puppygrad video list-devices`.
- [x] Document `puppygrad video capture-frame`.
- [x] Document `puppygrad video stream`.
- [x] Document camera permission requirements on macOS.
- [x] Document `puppygrad resnet --camera`.
- [x] Document the difference between ResNet whole-frame classification and YOLO-style object detection.
- [x] Document known limitations: camera backend support, CPU speed, no boxes/detection yet.

## Completion Criteria

- [x] `cargo fmt --check` passes.
- [x] `cargo check` passes.
- [x] Video module unit tests pass without requiring a camera.
- [x] `puppygrad video list-devices` prints available cameras.
- [x] `puppygrad video capture-frame --out /tmp/puppygrad-frame.jpg` writes a valid image from the default camera.
- [x] `puppygrad video capture-frame --device N --out /tmp/puppygrad-frame.jpg` works for a selected camera.
- [x] `puppygrad video stream --fps 1` captures frames continuously until Ctrl-C and reports stats.
- [x] `puppygrad resnet --camera --stream --fps 1 --top-k 3` classifies frames continuously when ResNet assets are available.
- [x] Existing `puppygrad resnet --image ...` still works.
- [x] Existing GPT-2 and Whisper commands still compile and run their smoke paths.
- [x] README documents video utilities, camera ResNet usage, and limitations.
