use super::weights::{FoldedConv2d, ResNet18Weights};
use crate::vision::cnn::{
    conv2d_nchw, global_avg_pool_nchw, linear, max_pool2d_nchw, relu_in_place,
    residual_add_in_place, NchwShape, NchwTensor,
};

#[derive(Clone, Debug)]
pub struct ResNet18Rust {
    weights: ResNet18Weights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResNet18ShapeTrace {
    pub stem: NchwShape,
    pub after_pool: NchwShape,
    pub stages: Vec<NchwShape>,
}

impl ResNet18Rust {
    pub fn new(weights: ResNet18Weights) -> Self {
        Self { weights }
    }

    pub fn logits(&self, input_chw: &[f32], height: usize, width: usize) -> Vec<f32> {
        let input = NchwTensor::new(
            NchwShape {
                n: 1,
                c: 3,
                h: height,
                w: width,
            },
            input_chw.to_vec(),
        );
        let (logits, _) = self.forward(input, false);
        logits
    }

    pub fn shape_trace(
        &self,
        input_chw: &[f32],
        height: usize,
        width: usize,
    ) -> ResNet18ShapeTrace {
        let input = NchwTensor::new(
            NchwShape {
                n: 1,
                c: 3,
                h: height,
                w: width,
            },
            input_chw.to_vec(),
        );
        let (_, trace) = self.forward(input, true);
        trace.unwrap()
    }

    fn forward(
        &self,
        input: NchwTensor,
        trace_shapes: bool,
    ) -> (Vec<f32>, Option<ResNet18ShapeTrace>) {
        let mut x = apply_conv(&input, &self.weights.stem);
        relu_in_place(&mut x.data);
        let stem_shape = x.shape;
        x = max_pool2d_nchw(&x, 3, 2, 1);
        let after_pool_shape = x.shape;
        let mut stage_shapes = Vec::new();

        for stage in &self.weights.stages {
            for block in stage {
                let residual = if let Some(downsample) = &block.downsample {
                    apply_conv(&x, downsample)
                } else {
                    x.clone()
                };
                let mut y = apply_conv(&x, &block.conv1);
                relu_in_place(&mut y.data);
                y = apply_conv(&y, &block.conv2);
                residual_add_in_place(&mut y, &residual);
                relu_in_place(&mut y.data);
                x = y;
            }
            if trace_shapes {
                stage_shapes.push(x.shape);
            }
        }

        let pooled = global_avg_pool_nchw(&x);
        let logits = linear(
            &pooled,
            &self.weights.fc_weight,
            &self.weights.fc_bias,
            self.weights.fc_bias.len(),
        );
        let trace = trace_shapes.then_some(ResNet18ShapeTrace {
            stem: stem_shape,
            after_pool: after_pool_shape,
            stages: stage_shapes,
        });
        (logits, trace)
    }
}

fn apply_conv(input: &NchwTensor, conv: &FoldedConv2d) -> NchwTensor {
    conv2d_nchw(
        input,
        &conv.weight,
        &conv.bias,
        conv.shape,
        conv.stride,
        conv.padding,
    )
}
