use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResNetVariant {
    ResNet18,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResNetBlockType {
    Basic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResNetConfig {
    pub variant: ResNetVariant,
    pub input_channels: usize,
    pub num_classes: usize,
    pub stem_kernel: usize,
    pub stem_stride: usize,
    pub stem_padding: usize,
    pub block_type: ResNetBlockType,
    pub stage_block_counts: Vec<usize>,
    pub stage_channels: Vec<usize>,
    pub stage_strides: Vec<usize>,
    pub batch_norm_epsilon: f32,
    pub resize_short_side: u32,
    pub crop_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ResNetConfig {
    pub fn resnet18_imagenet() -> Self {
        Self {
            variant: ResNetVariant::ResNet18,
            input_channels: 3,
            num_classes: 1000,
            stem_kernel: 7,
            stem_stride: 2,
            stem_padding: 3,
            block_type: ResNetBlockType::Basic,
            stage_block_counts: vec![2, 2, 2, 2],
            stage_channels: vec![64, 128, 256, 512],
            stage_strides: vec![1, 2, 2, 2],
            batch_norm_epsilon: 1e-5,
            resize_short_side: 256,
            crop_size: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

pub fn default_resnet18_dir() -> PathBuf {
    PathBuf::from("models/resnet18")
}
