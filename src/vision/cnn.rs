use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NchwShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl NchwShape {
    pub fn len(self) -> usize {
        self.n * self.c * self.h * self.w
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NchwTensor {
    pub shape: NchwShape,
    pub data: Vec<f32>,
}

impl NchwTensor {
    pub fn new(shape: NchwShape, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.len(), data.len());
        Self { shape, data }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dShape {
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
}

pub fn conv2d_nchw(
    input: &NchwTensor,
    weight: &[f32],
    bias: &[f32],
    weight_shape: Conv2dShape,
    stride: usize,
    padding: usize,
) -> NchwTensor {
    debug_assert_eq!(input.shape.c, weight_shape.in_channels);
    debug_assert_eq!(
        weight.len(),
        weight_shape.out_channels
            * weight_shape.in_channels
            * weight_shape.kernel_h
            * weight_shape.kernel_w
    );
    debug_assert_eq!(bias.len(), weight_shape.out_channels);
    let out_h = (input.shape.h + 2 * padding - weight_shape.kernel_h) / stride + 1;
    let out_w = (input.shape.w + 2 * padding - weight_shape.kernel_w) / stride + 1;
    let out_shape = NchwShape {
        n: input.shape.n,
        c: weight_shape.out_channels,
        h: out_h,
        w: out_w,
    };
    let mut out = vec![0.0; out_shape.len()];

    for n in 0..input.shape.n {
        for oc in 0..weight_shape.out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = bias[oc];
                    for ic in 0..weight_shape.in_channels {
                        for kh in 0..weight_shape.kernel_h {
                            let Some(ih) = spatial_index(oh, kh, stride, padding, input.shape.h)
                            else {
                                continue;
                            };
                            for kw in 0..weight_shape.kernel_w {
                                let Some(iw) =
                                    spatial_index(ow, kw, stride, padding, input.shape.w)
                                else {
                                    continue;
                                };
                                let input_index = ((n * input.shape.c + ic) * input.shape.h + ih)
                                    * input.shape.w
                                    + iw;
                                let weight_index = ((oc * weight_shape.in_channels + ic)
                                    * weight_shape.kernel_h
                                    + kh)
                                    * weight_shape.kernel_w
                                    + kw;
                                sum += input.data[input_index] * weight[weight_index];
                            }
                        }
                    }
                    let out_index = ((n * out_shape.c + oc) * out_h + oh) * out_w + ow;
                    out[out_index] = sum;
                }
            }
        }
    }

    NchwTensor::new(out_shape, out)
}

fn spatial_index(
    out_index: usize,
    kernel_index: usize,
    stride: usize,
    padding: usize,
    input_size: usize,
) -> Option<usize> {
    let raw = out_index * stride + kernel_index;
    if raw < padding {
        return None;
    }
    let input_index = raw - padding;
    (input_index < input_size).then_some(input_index)
}

pub fn relu_in_place(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

pub fn max_pool2d_nchw(
    input: &NchwTensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> NchwTensor {
    let out_h = (input.shape.h + 2 * padding - kernel) / stride + 1;
    let out_w = (input.shape.w + 2 * padding - kernel) / stride + 1;
    let out_shape = NchwShape {
        n: input.shape.n,
        c: input.shape.c,
        h: out_h,
        w: out_w,
    };
    let mut out = vec![f32::NEG_INFINITY; out_shape.len()];

    for n in 0..input.shape.n {
        for c in 0..input.shape.c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_value = f32::NEG_INFINITY;
                    for kh in 0..kernel {
                        let Some(ih) = spatial_index(oh, kh, stride, padding, input.shape.h) else {
                            continue;
                        };
                        for kw in 0..kernel {
                            let Some(iw) = spatial_index(ow, kw, stride, padding, input.shape.w)
                            else {
                                continue;
                            };
                            let index =
                                ((n * input.shape.c + c) * input.shape.h + ih) * input.shape.w + iw;
                            max_value = max_value.max(input.data[index]);
                        }
                    }
                    let out_index = ((n * out_shape.c + c) * out_h + oh) * out_w + ow;
                    out[out_index] = max_value;
                }
            }
        }
    }

    NchwTensor::new(out_shape, out)
}

pub fn global_avg_pool_nchw(input: &NchwTensor) -> Vec<f32> {
    let mut out = vec![0.0; input.shape.n * input.shape.c];
    let spatial = input.shape.h * input.shape.w;
    for n in 0..input.shape.n {
        for c in 0..input.shape.c {
            let mut sum = 0.0;
            for h in 0..input.shape.h {
                for w in 0..input.shape.w {
                    let index = ((n * input.shape.c + c) * input.shape.h + h) * input.shape.w + w;
                    sum += input.data[index];
                }
            }
            out[n * input.shape.c + c] = sum / spatial as f32;
        }
    }
    out
}

pub fn residual_add_in_place(dst: &mut NchwTensor, residual: &NchwTensor) {
    debug_assert_eq!(dst.shape, residual.shape);
    for (dst, residual) in dst.data.iter_mut().zip(residual.data.iter()) {
        *dst += residual;
    }
}

pub fn linear(input: &[f32], weight: &[f32], bias: &[f32], out_features: usize) -> Vec<f32> {
    let in_features = input.len();
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(bias.len(), out_features);
    let mut out = vec![0.0; out_features];
    for o in 0..out_features {
        let row = &weight[o * in_features..(o + 1) * in_features];
        let mut sum = bias[o];
        for i in 0..in_features {
            sum += input[i] * row[i];
        }
        out[o] = sum;
    }
    out
}

pub fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let mut sum = 0.0;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let exp = (*value - max).exp();
        sum += exp;
        out.push(exp);
    }
    for value in &mut out {
        *value /= sum;
    }
    out
}

pub fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed.truncate(k.min(indexed.len()));
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_nchw_supports_stride_and_padding() {
        let input = NchwTensor::new(
            NchwShape {
                n: 1,
                c: 1,
                h: 3,
                w: 3,
            },
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let weight = vec![1.0, 0.0, 0.0, -1.0];
        let out = conv2d_nchw(
            &input,
            &weight,
            &[0.5],
            Conv2dShape {
                out_channels: 1,
                in_channels: 1,
                kernel_h: 2,
                kernel_w: 2,
            },
            1,
            0,
        );
        assert_eq!(out.shape.h, 2);
        assert_eq!(out.shape.w, 2);
        assert_eq!(out.data, vec![-3.5, -3.5, -3.5, -3.5]);
    }

    #[test]
    fn max_pool_and_global_avg_pool_are_deterministic() {
        let input = NchwTensor::new(
            NchwShape {
                n: 1,
                c: 1,
                h: 4,
                w: 4,
            },
            (1..=16).map(|value| value as f32).collect(),
        );
        let pooled = max_pool2d_nchw(&input, 2, 2, 0);
        assert_eq!(pooled.shape.h, 2);
        assert_eq!(pooled.shape.w, 2);
        assert_eq!(pooled.data, vec![6.0, 8.0, 14.0, 16.0]);
        assert_eq!(global_avg_pool_nchw(&pooled), vec![11.0]);
    }

    #[test]
    fn residual_add_and_top_k_are_deterministic() {
        let mut dst = NchwTensor::new(
            NchwShape {
                n: 1,
                c: 1,
                h: 1,
                w: 3,
            },
            vec![1.0, 2.0, 3.0],
        );
        let residual = NchwTensor::new(dst.shape, vec![0.5, -1.0, 2.0]);
        residual_add_in_place(&mut dst, &residual);
        assert_eq!(dst.data, vec![1.5, 1.0, 5.0]);
        assert_eq!(top_k(&dst.data, 2), vec![(2, 5.0), (0, 1.5)]);
    }
}
