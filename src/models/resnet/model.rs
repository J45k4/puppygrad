use super::config::ResNetConfig;

#[derive(Clone, Debug, PartialEq)]
pub struct ResNetClassification {
    pub class_index: usize,
    pub label: String,
    pub probability: f32,
    pub logit: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResNetStageShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

pub fn resnet18_stage_shapes(config: &ResNetConfig) -> Vec<ResNetStageShape> {
    let mut height = config.crop_size as usize;
    let mut width = config.crop_size as usize;
    height = (height + 2 * config.stem_padding - config.stem_kernel) / config.stem_stride + 1;
    width = (width + 2 * config.stem_padding - config.stem_kernel) / config.stem_stride + 1;
    height = (height + 2 - 3) / 2 + 1;
    width = (width + 2 - 3) / 2 + 1;

    let mut shapes = Vec::with_capacity(config.stage_channels.len());
    for (stage_index, channels) in config.stage_channels.iter().copied().enumerate() {
        let stride = config.stage_strides[stage_index];
        if stride > 1 {
            height = (height + 2 - 3) / stride + 1;
            width = (width + 2 - 3) / stride + 1;
        }
        shapes.push(ResNetStageShape {
            channels,
            height,
            width,
        });
    }
    shapes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resnet::config::ResNetConfig;

    #[test]
    fn resnet18_stage_shapes_match_imagenet_input() {
        let shapes = resnet18_stage_shapes(&ResNetConfig::resnet18_imagenet());
        assert_eq!(
            shapes,
            vec![
                ResNetStageShape {
                    channels: 64,
                    height: 56,
                    width: 56
                },
                ResNetStageShape {
                    channels: 128,
                    height: 28,
                    width: 28
                },
                ResNetStageShape {
                    channels: 256,
                    height: 14,
                    width: 14
                },
                ResNetStageShape {
                    channels: 512,
                    height: 7,
                    width: 7
                },
            ]
        );
    }
}
