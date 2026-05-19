use image::RgbImage;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::{query, Camera};
use serde::Serialize;
use std::collections::VecDeque;
use std::error;
use std::fmt;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoError {
    Backend {
        operation: &'static str,
        message: String,
    },
    PermissionDenied {
        operation: &'static str,
        message: String,
    },
    MissingDevice {
        index: usize,
    },
    InvalidFrame {
        message: String,
    },
    CaptureTimeout {
        timeout: Duration,
    },
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Backend { operation, message } => {
                write!(f, "video {operation} failed: {message}")
            }
            VideoError::PermissionDenied { operation, message } => write!(
                f,
                "video {operation} was denied by the operating system: {message}"
            ),
            VideoError::MissingDevice { index } => write!(f, "camera device {index} was not found"),
            VideoError::InvalidFrame { message } => write!(f, "invalid video frame: {message}"),
            VideoError::CaptureTimeout { timeout } => {
                write!(
                    f,
                    "timed out after {:.1}s waiting for a camera frame",
                    timeout.as_secs_f32()
                )
            }
        }
    }
}

impl error::Error for VideoError {}

pub type VideoResult<T> = std::result::Result<T, VideoError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoDeviceInfo {
    pub index: usize,
    pub display_name: String,
    pub backend: String,
    pub is_default: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoPixelFormat {
    Rgb8,
    Rgba8,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub pixel_format: VideoPixelFormat,
    pub timestamp_millis: u128,
    pub capture_latency_millis: u128,
    #[serde(skip)]
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoCaptureOptions {
    pub device_index: Option<usize>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
    pub timeout: Duration,
}

impl Default for VideoCaptureOptions {
    fn default() -> Self {
        Self {
            device_index: None,
            width: None,
            height: None,
            fps: None,
            timeout: Duration::from_secs(5),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoDropPolicy {
    Oldest,
    Newest,
    Block,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VideoQueueStats {
    pub dropped_frames: u64,
}

#[derive(Clone, Debug)]
pub struct VideoFrameQueue {
    capacity: usize,
    policy: VideoDropPolicy,
    frames: VecDeque<VideoFrame>,
    stats: VideoQueueStats,
}

impl VideoFrameQueue {
    pub fn new(capacity: usize, policy: VideoDropPolicy) -> Self {
        Self {
            capacity: capacity.max(1),
            policy,
            frames: VecDeque::with_capacity(capacity.max(1)),
            stats: VideoQueueStats::default(),
        }
    }

    pub fn push(&mut self, frame: VideoFrame) -> bool {
        if self.frames.len() < self.capacity {
            self.frames.push_back(frame);
            return true;
        }

        match self.policy {
            VideoDropPolicy::Oldest => {
                self.frames.pop_front();
                self.frames.push_back(frame);
                self.stats.dropped_frames += 1;
                true
            }
            VideoDropPolicy::Newest => {
                self.stats.dropped_frames += 1;
                false
            }
            VideoDropPolicy::Block => false,
        }
    }

    pub fn pop(&mut self) -> Option<VideoFrame> {
        self.frames.pop_front()
    }

    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.capacity
    }

    pub fn dropped_frames(&self) -> u64 {
        self.stats.dropped_frames
    }
}

pub fn list_video_devices() -> VideoResult<Vec<VideoDeviceInfo>> {
    let cameras =
        query(ApiBackend::Auto).map_err(|source| map_backend_error("device query", source))?;
    Ok(cameras
        .into_iter()
        .enumerate()
        .map(|(ordinal, camera)| {
            let index = camera.index().as_index().unwrap_or(ordinal as u32) as usize;
            VideoDeviceInfo {
                index,
                display_name: camera.human_name().to_string(),
                backend: ApiBackend::Auto.to_string(),
                is_default: ordinal == 0,
            }
        })
        .collect())
}

pub fn capture_frame(options: VideoCaptureOptions) -> VideoResult<VideoFrame> {
    let mut camera = open_camera(options)?;
    camera
        .open_stream()
        .map_err(|source| map_backend_error("open stream", source))?;
    decode_next_frame(&mut camera, options.timeout)
}

pub fn open_camera(options: VideoCaptureOptions) -> VideoResult<Camera> {
    if let Some(index) = options.device_index {
        let devices = list_video_devices()?;
        if !devices.iter().any(|device| device.index == index) {
            return Err(VideoError::MissingDevice { index });
        }
    }

    let camera_index = CameraIndex::Index(options.device_index.unwrap_or(0) as u32);
    let request = requested_format(options);
    Camera::new(camera_index, request).map_err(|source| map_backend_error("camera open", source))
}

pub fn decode_next_frame(camera: &mut Camera, timeout: Duration) -> VideoResult<VideoFrame> {
    let start = std::time::Instant::now();
    loop {
        match camera.frame() {
            Ok(frame) => {
                let capture_latency_millis = start.elapsed().as_millis();
                let timestamp_millis = frame
                    .capture_timestamp()
                    .map(|timestamp| timestamp.as_millis())
                    .unwrap_or_else(now_millis);
                let decoded = frame
                    .decode_image::<RgbFormat>()
                    .map_err(|source| map_backend_error("RGB decode", source))?;
                return Ok(VideoFrame {
                    width: decoded.width(),
                    height: decoded.height(),
                    pixel_format: VideoPixelFormat::Rgb8,
                    timestamp_millis,
                    capture_latency_millis,
                    bytes: decoded.into_raw(),
                });
            }
            Err(source) if start.elapsed() < timeout => {
                let message = source.to_string();
                if is_permission_denied(&message) {
                    return Err(VideoError::PermissionDenied {
                        operation: "capture frame",
                        message,
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(source) => {
                let message = source.to_string();
                if is_permission_denied(&message) {
                    return Err(VideoError::PermissionDenied {
                        operation: "capture frame",
                        message,
                    });
                }
                return Err(VideoError::CaptureTimeout { timeout });
            }
        }
    }
}

pub fn video_frame_to_rgb8(frame: &VideoFrame) -> VideoResult<RgbImage> {
    match frame.pixel_format {
        VideoPixelFormat::Rgb8 => {
            RgbImage::from_raw(frame.width, frame.height, frame.bytes.clone()).ok_or_else(|| {
                VideoError::InvalidFrame {
                    message: format!(
                        "expected {} RGB bytes, got {}",
                        frame.width as usize * frame.height as usize * 3,
                        frame.bytes.len()
                    ),
                }
            })
        }
        VideoPixelFormat::Rgba8 => {
            let expected = frame.width as usize * frame.height as usize * 4;
            if frame.bytes.len() != expected {
                return Err(VideoError::InvalidFrame {
                    message: format!("expected {expected} RGBA bytes, got {}", frame.bytes.len()),
                });
            }
            let mut rgb = Vec::with_capacity(frame.width as usize * frame.height as usize * 3);
            for pixel in frame.bytes.chunks_exact(4) {
                rgb.extend_from_slice(&pixel[..3]);
            }
            RgbImage::from_raw(frame.width, frame.height, rgb).ok_or_else(|| {
                VideoError::InvalidFrame {
                    message: "failed to build RGB image from RGBA frame".to_string(),
                }
            })
        }
    }
}

pub fn save_video_frame(frame: &VideoFrame, path: &Path) -> VideoResult<()> {
    let image = video_frame_to_rgb8(frame)?;
    image.save(path).map_err(|source| VideoError::Backend {
        operation: "save frame",
        message: source.to_string(),
    })
}

fn requested_format(options: VideoCaptureOptions) -> RequestedFormat<'static> {
    match (options.width, options.height, options.fps) {
        (Some(width), Some(height), Some(fps)) => {
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
                Resolution::new(width, height),
                FrameFormat::MJPEG,
                fps,
            )))
        }
        (Some(width), Some(height), None) => RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::HighestResolution(Resolution::new(width, height)),
        ),
        (_, _, Some(_)) => {
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate)
        }
        _ => RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    }
}

fn map_backend_error(operation: &'static str, source: nokhwa::NokhwaError) -> VideoError {
    let message = source.to_string();
    if is_permission_denied(&message) {
        VideoError::PermissionDenied { operation, message }
    } else {
        VideoError::Backend { operation, message }
    }
}

fn is_permission_denied(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("permission") || lower.contains("denied") || lower.contains("authorized")
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(index: u8) -> VideoFrame {
        VideoFrame {
            width: 1,
            height: 1,
            pixel_format: VideoPixelFormat::Rgb8,
            timestamp_millis: index as u128,
            capture_latency_millis: 0,
            bytes: vec![index, 0, 0],
        }
    }

    #[test]
    fn rgb_frame_converts_to_rgb_image() -> VideoResult<()> {
        let image = video_frame_to_rgb8(&VideoFrame {
            width: 2,
            height: 1,
            pixel_format: VideoPixelFormat::Rgb8,
            timestamp_millis: 7,
            capture_latency_millis: 0,
            bytes: vec![1, 2, 3, 4, 5, 6],
        })?;

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.as_raw(), &[1, 2, 3, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn rgba_frame_drops_alpha() -> VideoResult<()> {
        let image = video_frame_to_rgb8(&VideoFrame {
            width: 2,
            height: 1,
            pixel_format: VideoPixelFormat::Rgba8,
            timestamp_millis: 7,
            capture_latency_millis: 0,
            bytes: vec![1, 2, 3, 255, 4, 5, 6, 128],
        })?;

        assert_eq!(image.as_raw(), &[1, 2, 3, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn queue_drops_oldest_by_default_policy() {
        let mut queue = VideoFrameQueue::new(2, VideoDropPolicy::Oldest);
        assert!(queue.push(frame(1)));
        assert!(queue.push(frame(2)));
        assert!(queue.push(frame(3)));

        assert_eq!(queue.dropped_frames(), 1);
        assert_eq!(queue.pop().unwrap().timestamp_millis, 2);
        assert_eq!(queue.pop().unwrap().timestamp_millis, 3);
    }

    #[test]
    fn queue_drops_newest_when_configured() {
        let mut queue = VideoFrameQueue::new(1, VideoDropPolicy::Newest);
        assert!(queue.push(frame(1)));
        assert!(!queue.push(frame(2)));

        assert_eq!(queue.dropped_frames(), 1);
        assert_eq!(queue.pop().unwrap().timestamp_millis, 1);
    }

    #[test]
    fn queue_block_policy_refuses_when_full() {
        let mut queue = VideoFrameQueue::new(1, VideoDropPolicy::Block);
        assert!(queue.push(frame(1)));
        assert!(!queue.push(frame(2)));

        assert_eq!(queue.dropped_frames(), 0);
        assert_eq!(queue.pop().unwrap().timestamp_millis, 1);
    }
}
