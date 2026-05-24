use std::path::Path;

use image::{ImageBuffer, ImageFormat, RgbImage};

use super::vae::validate_decoded_rgb_shape;
use super::{Result, SdTensor, StableDiffusionError, StableDiffusionOutputFormat};

pub fn rgb_tensor_to_image(tensor: &SdTensor, width: u32, height: u32) -> Result<RgbImage> {
    validate_decoded_rgb_shape(tensor, width, height)?;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut bytes = vec![0u8; width_usize * height_usize * 3];
    for y in 0..height_usize {
        for x in 0..width_usize {
            let pixel = (y * width_usize + x) * 3;
            for channel in 0..3 {
                let value = tensor.data()[((channel * height_usize + y) * width_usize) + x];
                bytes[pixel + channel] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    ImageBuffer::from_raw(width, height, bytes).ok_or_else(|| {
        StableDiffusionError::Image(format!(
            "failed to build RGB image buffer for {width}x{height}"
        ))
    })
}

pub fn save_rgb_tensor_image(
    tensor: &SdTensor,
    path: &Path,
    width: u32,
    height: u32,
    format: StableDiffusionOutputFormat,
) -> Result<()> {
    let image = rgb_tensor_to_image(tensor, width, height)?;
    let format = match format {
        StableDiffusionOutputFormat::Png => ImageFormat::Png,
        StableDiffusionOutputFormat::Jpeg => ImageFormat::Jpeg,
    };
    image
        .save_with_format(path, format)
        .map_err(|err| StableDiffusionError::Image(err.to_string()))?;
    Ok(())
}

pub fn diffusers_decoded_to_rgb(decoded: &SdTensor) -> Result<SdTensor> {
    if decoded.rank() != 4 || decoded.shape()[0] != 1 || decoded.shape()[1] != 3 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "decoded image tensor must have shape [1, 3, H, W], got {:?}",
            decoded.shape()
        )));
    }
    decoded.scale(0.5)?.add_scalar(0.5)?.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_nchw_rgb_tensor_to_image_bytes() -> Result<()> {
        let tensor = SdTensor::new(
            [1, 3, 2, 2],
            vec![
                0.0, 0.5, 1.0, 0.25, 1.0, 0.5, 0.0, 0.25, 0.0, 0.25, 0.5, 1.0,
            ],
        )?;

        let image = rgb_tensor_to_image(&tensor, 2, 2)?;

        assert_eq!(image.dimensions(), (2, 2));
        assert_eq!(
            image.as_raw(),
            &[0, 255, 0, 128, 128, 64, 255, 0, 128, 64, 64, 255]
        );
        Ok(())
    }

    #[test]
    fn maps_diffusers_minus_one_to_one_range_to_rgb_range() -> Result<()> {
        let decoded = SdTensor::new([1, 3, 1, 1], vec![-1.0, 0.0, 1.0])?;

        let rgb = diffusers_decoded_to_rgb(&decoded)?;

        assert_eq!(rgb.data(), &[0.0, 0.5, 1.0]);
        Ok(())
    }
}
