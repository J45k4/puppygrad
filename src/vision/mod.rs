pub mod cnn;

use image::imageops::{crop_imm, resize, FilterType};
use image::{ImageError, ImageReader, RgbImage};
use std::error;
use std::fmt;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct RgbImageF32 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChwImage {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum VisionError {
    OpenImage {
        path: String,
        source: std::io::Error,
    },
    DecodeImage {
        path: String,
        source: ImageError,
    },
    InvalidCrop {
        image_width: usize,
        image_height: usize,
        crop_width: usize,
        crop_height: usize,
    },
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VisionError::OpenImage { path, source } => {
                write!(f, "failed to open image {path}: {source}")
            }
            VisionError::DecodeImage { path, source } => {
                write!(f, "failed to decode image {path}: {source}")
            }
            VisionError::InvalidCrop {
                image_width,
                image_height,
                crop_width,
                crop_height,
            } => write!(
                f,
                "cannot center crop {crop_width}x{crop_height} from image {image_width}x{image_height}"
            ),
        }
    }
}

impl error::Error for VisionError {}

pub type Result<T> = std::result::Result<T, VisionError>;

pub fn load_rgb_image(path: &Path) -> Result<RgbImageF32> {
    let path_display = path.display().to_string();
    let image = ImageReader::open(path)
        .map_err(|source| VisionError::OpenImage {
            path: path_display.clone(),
            source,
        })?
        .decode()
        .map_err(|source| VisionError::DecodeImage {
            path: path_display,
            source,
        })?
        .to_rgb8();
    Ok(rgb8_to_hwc_f32(&image))
}

pub fn load_rgb8(path: &Path) -> Result<RgbImage> {
    let path_display = path.display().to_string();
    ImageReader::open(path)
        .map_err(|source| VisionError::OpenImage {
            path: path_display.clone(),
            source,
        })?
        .decode()
        .map(|image| image.to_rgb8())
        .map_err(|source| VisionError::DecodeImage {
            path: path_display,
            source,
        })
}

pub fn resize_shortest_side(image: &RgbImage, shortest_side: u32, filter: FilterType) -> RgbImage {
    let width = image.width();
    let height = image.height();
    if width <= height {
        let new_height = ((height as f32 * shortest_side as f32) / width as f32).round() as u32;
        resize(image, shortest_side, new_height, filter)
    } else {
        let new_width = ((width as f32 * shortest_side as f32) / height as f32).round() as u32;
        resize(image, new_width, shortest_side, filter)
    }
}

pub fn center_crop(image: &RgbImage, crop_width: u32, crop_height: u32) -> Result<RgbImage> {
    let width = image.width();
    let height = image.height();
    if crop_width > width || crop_height > height {
        return Err(VisionError::InvalidCrop {
            image_width: width as usize,
            image_height: height as usize,
            crop_width: crop_width as usize,
            crop_height: crop_height as usize,
        });
    }
    let x = (width - crop_width) / 2;
    let y = (height - crop_height) / 2;
    Ok(crop_imm(image, x, y, crop_width, crop_height).to_image())
}

pub fn rgb8_to_hwc_f32(image: &RgbImage) -> RgbImageF32 {
    let mut data = Vec::with_capacity(image.width() as usize * image.height() as usize * 3);
    for pixel in image.pixels() {
        data.push(pixel[0] as f32 / 255.0);
        data.push(pixel[1] as f32 / 255.0);
        data.push(pixel[2] as f32 / 255.0);
    }
    RgbImageF32 {
        width: image.width() as usize,
        height: image.height() as usize,
        data,
    }
}

pub fn hwc_to_chw(image: &RgbImageF32) -> ChwImage {
    let mut data = vec![0.0; image.width * image.height * 3];
    for y in 0..image.height {
        for x in 0..image.width {
            for c in 0..3 {
                data[c * image.height * image.width + y * image.width + x] =
                    image.data[(y * image.width + x) * 3 + c];
            }
        }
    }
    ChwImage {
        channels: 3,
        height: image.height,
        width: image.width,
        data,
    }
}

pub fn chw_to_nchw(chw: &ChwImage) -> Vec<f32> {
    chw.data.clone()
}

pub fn normalize_chw_in_place(chw: &mut ChwImage, mean: [f32; 3], std: [f32; 3]) {
    debug_assert_eq!(chw.channels, 3);
    let plane = chw.height * chw.width;
    for c in 0..3 {
        for value in &mut chw.data[c * plane..(c + 1) * plane] {
            *value = (*value - mean[c]) / std[c];
        }
    }
}

pub fn preprocess_rgb8_to_normalized_chw(
    image: &RgbImage,
    resize_short_side: u32,
    crop_size: u32,
    mean: [f32; 3],
    std: [f32; 3],
    filter: FilterType,
) -> Result<ChwImage> {
    let resized = resize_shortest_side(image, resize_short_side, filter);
    let cropped = center_crop(&resized, crop_size, crop_size)?;
    let hwc = rgb8_to_hwc_f32(&cropped);
    let mut chw = hwc_to_chw(&hwc);
    normalize_chw_in_place(&mut chw, mean, std);
    Ok(chw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn preprocessing_converts_rgb_to_normalized_224_chw() -> Result<()> {
        let mut image = RgbImage::new(300, 260);
        for y in 0..260 {
            for x in 0..300 {
                image.put_pixel(x, y, Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
            }
        }

        let chw = preprocess_rgb8_to_normalized_chw(
            &image,
            256,
            224,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            FilterType::Triangle,
        )?;

        assert_eq!(chw.channels, 3);
        assert_eq!(chw.height, 224);
        assert_eq!(chw.width, 224);
        assert_eq!(chw.data.len(), 3 * 224 * 224);
        assert!(chw.data.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn layout_helpers_convert_hwc_to_chw_and_nchw() {
        let image = RgbImageF32 {
            width: 2,
            height: 1,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let chw = hwc_to_chw(&image);
        assert_eq!(chw.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(chw_to_nchw(&chw), chw.data);
    }
}
