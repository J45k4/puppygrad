use std::thread;

use gemm::Parallelism;

use super::{Result, StableDiffusionError};

const PARALLEL_MATMUL_THRESHOLD: usize = 1_000_000;
const PARALLEL_GROUP_NORM_THRESHOLD: usize = 64 * 1024;
const PARALLEL_LAYER_NORM_THRESHOLD: usize = 64 * 1024;
const PARALLEL_ATTENTION_THRESHOLD: usize = 1_000_000;
const PARALLEL_ELEMENTWISE_THRESHOLD: usize = 256 * 1024;
const GEMM_LINEAR_THRESHOLD: usize = 4_000_000;
const GEMM_CONV1X1_THRESHOLD: usize = 4_000_000;
const GEMM_CONV3X3_THRESHOLD: usize = 64_000_000;
const PACKED_SIDE_TAP_MAX_SPATIAL: usize = 4_096;

#[derive(Clone, Debug, PartialEq)]
pub struct SdTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dOptions {
    pub stride: usize,
    pub padding: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub stddev: f32,
    pub rms: f32,
}

impl SdTensor {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        let expected = tensor_len(&shape)?;
        if expected != data.len() {
            return Err(StableDiffusionError::InvalidInput(format!(
                "tensor shape {:?} expects {expected} values, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let len = tensor_len(&shape)?;
        Self::new(shape, vec![0.0; len])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn require_shape(&self, expected: &[usize]) -> Result<()> {
        if self.shape != expected {
            return Err(StableDiffusionError::InvalidInput(format!(
                "tensor shape {:?} does not match expected {:?}",
                self.shape, expected
            )));
        }
        Ok(())
    }

    pub fn into_shape(mut self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let expected_len = shape.iter().product::<usize>();
        if expected_len != self.data.len() {
            return Err(StableDiffusionError::InvalidInput(format!(
                "cannot reshape tensor with {} values to {:?}",
                self.data.len(),
                shape
            )));
        }
        self.shape = shape;
        Ok(self)
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    pub fn stats(&self) -> Result<TensorStats> {
        tensor_stats(&self.data)
    }

    pub fn selected_slice(&self, offset: usize, len: usize) -> Result<Vec<f32>> {
        let end = offset.checked_add(len).ok_or_else(|| {
            StableDiffusionError::InvalidInput("slice range overflow".to_string())
        })?;
        if end > self.data.len() {
            return Err(StableDiffusionError::InvalidInput(format!(
                "slice {offset}..{end} exceeds tensor length {}",
                self.data.len()
            )));
        }
        Ok(self.data[offset..end].to_vec())
    }

    pub fn map_binary(&self, rhs: &Self, op: impl Fn(f32, f32) -> f32) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(StableDiffusionError::InvalidInput(format!(
                "cannot combine shapes {:?} and {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(left, right)| op(*left, *right))
            .collect();
        Self::new(self.shape.clone(), data)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        broadcast_binary(self, rhs, |left, right| left + right)
    }

    pub fn add_same_shape_in_place(&mut self, rhs: &Self) -> Result<()> {
        if self.shape != rhs.shape {
            return Err(StableDiffusionError::InvalidInput(format!(
                "cannot add shapes {:?} and {:?} in place",
                self.shape, rhs.shape
            )));
        }
        let workers = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(self.data.len().max(1));
        if workers > 1 && self.data.len() >= PARALLEL_ELEMENTWISE_THRESHOLD {
            let values_per_chunk = self.data.len().div_ceil(workers);
            thread::scope(|scope| {
                for (chunk_index, out_chunk) in self.data.chunks_mut(values_per_chunk).enumerate() {
                    let first = chunk_index * values_per_chunk;
                    let rhs_chunk = &rhs.data[first..first + out_chunk.len()];
                    scope.spawn(move || {
                        for (dst, src) in out_chunk.iter_mut().zip(rhs_chunk.iter().copied()) {
                            *dst += src;
                        }
                    });
                }
            });
        } else {
            for (dst, src) in self.data.iter_mut().zip(rhs.data.iter().copied()) {
                *dst += src;
            }
        }
        Ok(())
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        broadcast_binary(self, rhs, |left, right| left - right)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        broadcast_binary(self, rhs, |left, right| left * right)
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        broadcast_binary(self, rhs, |left, right| left / right)
    }

    pub fn scale(&self, value: f32) -> Result<Self> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|element| element * value).collect(),
        )
    }

    pub fn add_scalar(&self, value: f32) -> Result<Self> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|element| element + value).collect(),
        )
    }

    pub fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        if min > max {
            return Err(StableDiffusionError::InvalidInput(
                "clamp min must be <= max".to_string(),
            ));
        }
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .map(|element| element.clamp(min, max))
                .collect(),
        )
    }

    pub fn silu(&self) -> Result<Self> {
        self.silu_impl(true)
    }

    fn silu_impl(&self, allow_parallel: bool) -> Result<Self> {
        unary_map(&self.shape, &self.data, allow_parallel, |value| {
            value / (1.0 + (-value).exp())
        })
    }

    pub fn gelu(&self) -> Result<Self> {
        self.gelu_impl(true)
    }

    fn gelu_impl(&self, allow_parallel: bool) -> Result<Self> {
        unary_map(&self.shape, &self.data, allow_parallel, |value| {
            0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value * value * value)).tanh())
        })
    }
}

fn unary_map(
    shape: &[usize],
    data: &[f32],
    allow_parallel: bool,
    op: impl Fn(f32) -> f32 + Copy + Send + Sync,
) -> Result<SdTensor> {
    let mut out = vec![0.0; data.len()];
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(data.len().max(1));

    if allow_parallel && workers > 1 && data.len() >= PARALLEL_ELEMENTWISE_THRESHOLD {
        let values_per_chunk = data.len().div_ceil(workers);
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let first = chunk_index * values_per_chunk;
                let input_chunk = &data[first..first + out_chunk.len()];
                scope.spawn(move || {
                    for (dst, src) in out_chunk.iter_mut().zip(input_chunk.iter().copied()) {
                        *dst = op(src);
                    }
                });
            }
        });
    } else {
        for (dst, src) in out.iter_mut().zip(data.iter().copied()) {
            *dst = op(src);
        }
    }
    SdTensor::new(shape.to_vec(), out)
}

pub fn tensor_stats(data: &[f32]) -> Result<TensorStats> {
    if data.is_empty() {
        return Err(StableDiffusionError::InvalidInput(
            "cannot compute statistics for an empty tensor".to_string(),
        ));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(StableDiffusionError::InvalidInput(
            "tensor contains non-finite values".to_string(),
        ));
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut square_sum = 0.0f64;
    for value in data {
        min = min.min(*value);
        max = max.max(*value);
        sum += f64::from(*value);
        square_sum += f64::from(*value) * f64::from(*value);
    }
    let mean = (sum / data.len() as f64) as f32;
    let rms = (square_sum / data.len() as f64).sqrt() as f32;
    let variance = data
        .iter()
        .map(|value| {
            let delta = f64::from(*value) - f64::from(mean);
            delta * delta
        })
        .sum::<f64>()
        / data.len() as f64;
    Ok(TensorStats {
        min,
        max,
        mean,
        stddev: variance.sqrt() as f32,
        rms,
    })
}

pub fn broadcast_binary(
    left: &SdTensor,
    right: &SdTensor,
    op: impl Fn(f32, f32) -> f32 + Copy + Send + Sync,
) -> Result<SdTensor> {
    if left.shape == right.shape {
        return binary_map_same_shape(left, right, true, op);
    }
    if right.data.len() == 1 && right.shape.iter().all(|dim| *dim == 1) {
        return unary_map(&left.shape, &left.data, true, |value| {
            op(value, right.data[0])
        });
    }
    if left.data.len() == 1 && left.shape.iter().all(|dim| *dim == 1) {
        return unary_map(&right.shape, &right.data, true, |value| {
            op(left.data[0], value)
        });
    }
    let shape = broadcast_shape(&left.shape, &right.shape)?;
    let len = tensor_len(&shape)?;
    let mut data = Vec::with_capacity(len);
    for index in 0..len {
        let out_indices = linear_to_indices(index, &shape);
        let left_index = broadcast_linear_index(&out_indices, &shape, &left.shape);
        let right_index = broadcast_linear_index(&out_indices, &shape, &right.shape);
        data.push(op(left.data[left_index], right.data[right_index]));
    }
    SdTensor::new(shape, data)
}

fn binary_map_same_shape(
    left: &SdTensor,
    right: &SdTensor,
    allow_parallel: bool,
    op: impl Fn(f32, f32) -> f32 + Copy + Send + Sync,
) -> Result<SdTensor> {
    let mut out = vec![0.0; left.data.len()];
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(out.len().max(1));

    if allow_parallel && workers > 1 && out.len() >= PARALLEL_ELEMENTWISE_THRESHOLD {
        let values_per_chunk = out.len().div_ceil(workers);
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let first = chunk_index * values_per_chunk;
                let left_chunk = &left.data[first..first + out_chunk.len()];
                let right_chunk = &right.data[first..first + out_chunk.len()];
                scope.spawn(move || {
                    for ((dst, left), right) in out_chunk
                        .iter_mut()
                        .zip(left_chunk.iter().copied())
                        .zip(right_chunk.iter().copied())
                    {
                        *dst = op(left, right);
                    }
                });
            }
        });
    } else {
        for ((dst, left), right) in out
            .iter_mut()
            .zip(left.data.iter().copied())
            .zip(right.data.iter().copied())
        {
            *dst = op(left, right);
        }
    }
    SdTensor::new(left.shape.clone(), out)
}

pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut out = vec![1; rank];
    for offset in 0..rank {
        let left_dim = dim_from_right(left, offset).unwrap_or(1);
        let right_dim = dim_from_right(right, offset).unwrap_or(1);
        let dim = match (left_dim, right_dim) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (a, b) => {
                return Err(StableDiffusionError::InvalidInput(format!(
                    "cannot broadcast dimensions {a} and {b} for shapes {left:?} and {right:?}"
                )))
            }
        };
        out[rank - 1 - offset] = dim;
    }
    Ok(out)
}

pub fn concat_tensors(axis: usize, tensors: &[SdTensor]) -> Result<SdTensor> {
    let first = tensors.first().ok_or_else(|| {
        StableDiffusionError::InvalidInput("concat requires at least one tensor".to_string())
    })?;
    if axis >= first.shape.len() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "concat axis {axis} is out of range for shape {:?}",
            first.shape
        )));
    }
    if tensors
        .iter()
        .any(|tensor| tensor.shape.len() != first.shape.len())
    {
        return Err(StableDiffusionError::InvalidInput(
            "concat tensors must have the same rank".to_string(),
        ));
    }
    let mut shape = first.shape.clone();
    shape[axis] = tensors.iter().map(|tensor| tensor.shape[axis]).sum();
    for tensor in tensors {
        for (dim_index, (actual, expected)) in
            tensor.shape.iter().zip(first.shape.iter()).enumerate()
        {
            if dim_index != axis && actual != expected {
                return Err(StableDiffusionError::InvalidInput(format!(
                    "concat dimension {dim_index} mismatch: got {actual}, expected {expected}"
                )));
            }
        }
    }

    let inner: usize = first.shape[axis + 1..].iter().product();
    let outer: usize = first.shape[..axis].iter().product();
    let output_axis_len = shape[axis];
    let mut data = vec![0.0; tensor_len(&shape)?];
    for outer_index in 0..outer {
        let mut output_offset = (outer_index * output_axis_len) * inner;
        for tensor in tensors {
            let values = tensor.shape[axis] * inner;
            let input_offset = (outer_index * tensor.shape[axis]) * inner;
            data[output_offset..output_offset + values]
                .copy_from_slice(&tensor.data[input_offset..input_offset + values]);
            output_offset += values;
        }
    }
    SdTensor::new(shape, data)
}

pub fn split_tensor(axis: usize, tensor: &SdTensor, sizes: &[usize]) -> Result<Vec<SdTensor>> {
    if axis >= tensor.rank() {
        return Err(StableDiffusionError::InvalidInput(format!(
            "split axis {axis} is out of range for shape {:?}",
            tensor.shape
        )));
    }
    if sizes.iter().sum::<usize>() != tensor.shape[axis] {
        return Err(StableDiffusionError::InvalidInput(format!(
            "split sizes {:?} do not sum to axis {} length {}",
            sizes, axis, tensor.shape[axis]
        )));
    }

    let mut out = Vec::with_capacity(sizes.len());
    let mut axis_offset = 0;
    for size in sizes {
        let mut shape = tensor.shape.clone();
        shape[axis] = *size;
        let mut data = vec![0.0; tensor_len(&shape)?];
        for (dst_index, dst) in data.iter_mut().enumerate() {
            let mut src_indices = linear_to_indices(dst_index, &shape);
            src_indices[axis] += axis_offset;
            *dst = tensor.data[indices_to_linear(&src_indices, &tensor.shape)];
        }
        out.push(SdTensor::new(shape, data)?);
        axis_offset += size;
    }
    Ok(out)
}

pub fn channel_affine_nchw(input: &SdTensor, scale: &[f32], bias: &[f32]) -> Result<SdTensor> {
    let [n, c, h, w] = shape4(input, "channel affine")?;
    if scale.len() != c || bias.len() != c {
        return Err(StableDiffusionError::InvalidInput(format!(
            "channel affine expected {c} scale/bias values, got {} and {}",
            scale.len(),
            bias.len()
        )));
    }
    let mut data = input.data.clone();
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let index = nchw_index(batch, channel, y, x, c, h, w);
                    data[index] = data[index] * scale[channel] + bias[channel];
                }
            }
        }
    }
    SdTensor::new(input.shape.clone(), data)
}

pub fn upsample_nearest2d_nchw(input: &SdTensor, scale: usize) -> Result<SdTensor> {
    let [n, c, h, w] = shape4(input, "nearest upsample")?;
    if scale == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "nearest upsample scale must be > 0".to_string(),
        ));
    }
    if scale == 2 {
        return upsample_nearest2d_scale2_nchw(input, n, c, h, w);
    }
    let out_h = h * scale;
    let out_w = w * scale;
    let mut out = vec![0.0; n * c * out_h * out_w];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..out_h {
                for x in 0..out_w {
                    let src = nchw_index(batch, channel, y / scale, x / scale, c, h, w);
                    let dst = nchw_index(batch, channel, y, x, c, out_h, out_w);
                    out[dst] = input.data[src];
                }
            }
        }
    }
    SdTensor::new([n, c, out_h, out_w], out)
}

fn upsample_nearest2d_scale2_nchw(
    input: &SdTensor,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Result<SdTensor> {
    let out_h = h * 2;
    let out_w = w * 2;
    let input_plane_len = h * w;
    let output_plane_len = out_h * out_w;
    let mut out = vec![0.0; n * c * output_plane_len];
    for plane in 0..n * c {
        let input_plane = &input.data[plane * input_plane_len..(plane + 1) * input_plane_len];
        let output_plane = &mut out[plane * output_plane_len..(plane + 1) * output_plane_len];
        for y in 0..h {
            let input_row = &input_plane[y * w..(y + 1) * w];
            let row0_start = (y * 2) * out_w;
            let row1_start = row0_start + out_w;
            let (before_row1, from_row1) = output_plane.split_at_mut(row1_start);
            let row0 = &mut before_row1[row0_start..row0_start + out_w];
            let row1 = &mut from_row1[..out_w];
            for (x, value) in input_row.iter().copied().enumerate() {
                let out_x = x * 2;
                row0[out_x] = value;
                row0[out_x + 1] = value;
                row1[out_x] = value;
                row1[out_x + 1] = value;
            }
        }
    }
    SdTensor::new([n, c, out_h, out_w], out)
}

pub fn downsample_nearest2d_nchw(input: &SdTensor, stride: usize) -> Result<SdTensor> {
    let [n, c, h, w] = shape4(input, "nearest downsample")?;
    if stride == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "nearest downsample stride must be > 0".to_string(),
        ));
    }
    let out_h = h / stride;
    let out_w = w / stride;
    if out_h == 0 || out_w == 0 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "nearest downsample stride {stride} is too large for {h}x{w}"
        )));
    }
    let mut out = vec![0.0; n * c * out_h * out_w];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..out_h {
                for x in 0..out_w {
                    let src = nchw_index(batch, channel, y * stride, x * stride, c, h, w);
                    let dst = nchw_index(batch, channel, y, x, c, out_h, out_w);
                    out[dst] = input.data[src];
                }
            }
        }
    }
    SdTensor::new([n, c, out_h, out_w], out)
}

pub fn conv2d_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: Option<&[f32]>,
    options: Conv2dOptions,
) -> Result<SdTensor> {
    conv2d_nchw_impl(input, weight, bias, options, true)
}

fn conv2d_nchw_impl(
    input: &SdTensor,
    weight: &SdTensor,
    bias: Option<&[f32]>,
    options: Conv2dOptions,
    allow_parallel: bool,
) -> Result<SdTensor> {
    let [n, in_channels, in_h, in_w] = shape4(input, "conv2d input")?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight, "conv2d weight")?;
    if in_channels != weight_in_channels {
        return Err(StableDiffusionError::InvalidInput(format!(
            "conv2d input channels {in_channels} do not match weight channels {weight_in_channels}"
        )));
    }
    if options.stride == 0 {
        return Err(StableDiffusionError::InvalidInput(
            "conv2d stride must be > 0".to_string(),
        ));
    }
    let bias = match bias {
        Some(bias) if bias.len() == out_channels => bias.to_vec(),
        Some(bias) => {
            return Err(StableDiffusionError::InvalidInput(format!(
                "conv2d bias length {} does not match output channels {out_channels}",
                bias.len()
            )));
        }
        None => vec![0.0; out_channels],
    };
    let padded_h = in_h + options.padding * 2;
    let padded_w = in_w + options.padding * 2;
    if padded_h < kernel_h || padded_w < kernel_w {
        return Err(StableDiffusionError::InvalidInput(
            "conv2d kernel is larger than padded input".to_string(),
        ));
    }
    let out_h = (padded_h - kernel_h) / options.stride + 1;
    let out_w = (padded_w - kernel_w) / options.stride + 1;
    if kernel_h == 1 && kernel_w == 1 && options.stride == 1 && options.padding == 0 {
        return conv2d_1x1_nchw(input, weight, &bias, allow_parallel);
    }
    if kernel_h == 3 && kernel_w == 3 && options.stride == 1 && options.padding == 1 {
        return conv2d_3x3_pad1_nchw(input, weight, &bias, allow_parallel);
    }
    let mut out = vec![0.0; n * out_channels * out_h * out_w];

    let output_planes = n * out_channels;
    let plane_len = out_h * out_w;
    let estimated_mul_adds = output_planes
        .saturating_mul(plane_len)
        .saturating_mul(in_channels)
        .saturating_mul(kernel_h)
        .saturating_mul(kernel_w);
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(output_planes);
    const PARALLEL_CONV2D_THRESHOLD: usize = 1_000_000;

    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_CONV2D_THRESHOLD {
        let planes_per_chunk = output_planes.div_ceil(workers);
        let values_per_chunk = planes_per_chunk * plane_len;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let input_data = &input.data;
                let weight_data = &weight.data;
                let bias = &bias;
                let first_plane = chunk_index * planes_per_chunk;
                scope.spawn(move || {
                    for (local_plane, out_plane) in out_chunk.chunks_mut(plane_len).enumerate() {
                        let plane = first_plane + local_plane;
                        let batch = plane / out_channels;
                        let out_channel = plane % out_channels;
                        fill_conv2d_plane(
                            input_data,
                            weight_data,
                            bias,
                            batch,
                            out_channel,
                            in_channels,
                            in_h,
                            in_w,
                            kernel_h,
                            kernel_w,
                            out_h,
                            out_w,
                            options,
                            out_plane,
                        );
                    }
                });
            }
        });
    } else {
        for batch in 0..n {
            for out_channel in 0..out_channels {
                let plane_start = nchw_index(batch, out_channel, 0, 0, out_channels, out_h, out_w);
                fill_conv2d_plane(
                    &input.data,
                    &weight.data,
                    &bias,
                    batch,
                    out_channel,
                    in_channels,
                    in_h,
                    in_w,
                    kernel_h,
                    kernel_w,
                    out_h,
                    out_w,
                    options,
                    &mut out[plane_start..plane_start + plane_len],
                );
            }
        }
    }
    SdTensor::new([n, out_channels, out_h, out_w], out)
}

fn conv2d_1x1_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: &[f32],
    allow_parallel: bool,
) -> Result<SdTensor> {
    let [n, in_channels, h, w] = shape4(input, "conv2d input")?;
    let [out_channels, _, _, _] = shape4(weight, "conv2d weight")?;
    let spatial = h * w;
    let estimated_mul_adds = n
        .saturating_mul(out_channels)
        .saturating_mul(spatial)
        .saturating_mul(in_channels);
    let available_workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if allow_parallel && available_workers > 1 && estimated_mul_adds >= GEMM_CONV1X1_THRESHOLD {
        return conv2d_1x1_gemm_nchw(input, weight, bias, available_workers);
    }
    conv2d_1x1_direct_nchw(input, weight, bias, allow_parallel)
}

fn conv2d_1x1_direct_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: &[f32],
    allow_parallel: bool,
) -> Result<SdTensor> {
    let [n, in_channels, h, w] = shape4(input, "conv2d input")?;
    let [out_channels, _, _, _] = shape4(weight, "conv2d weight")?;
    let spatial = h * w;
    let output_planes = n * out_channels;
    let mut out = vec![0.0; output_planes * spatial];
    let estimated_mul_adds = output_planes
        .saturating_mul(spatial)
        .saturating_mul(in_channels);
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(output_planes);
    const PARALLEL_CONV2D_THRESHOLD: usize = 1_000_000;

    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_CONV2D_THRESHOLD {
        let planes_per_chunk = output_planes.div_ceil(workers);
        let values_per_chunk = planes_per_chunk * spatial;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let input_data = &input.data;
                let weight_data = &weight.data;
                let first_plane = chunk_index * planes_per_chunk;
                scope.spawn(move || {
                    for (local_plane, out_plane) in out_chunk.chunks_mut(spatial).enumerate() {
                        let plane = first_plane + local_plane;
                        let batch = plane / out_channels;
                        let out_channel = plane % out_channels;
                        fill_conv2d_1x1_plane(
                            input_data,
                            weight_data,
                            bias,
                            batch,
                            out_channel,
                            in_channels,
                            h,
                            w,
                            out_plane,
                        );
                    }
                });
            }
        });
    } else {
        for batch in 0..n {
            for out_channel in 0..out_channels {
                let plane_start = (batch * out_channels + out_channel) * spatial;
                fill_conv2d_1x1_plane(
                    &input.data,
                    &weight.data,
                    bias,
                    batch,
                    out_channel,
                    in_channels,
                    h,
                    w,
                    &mut out[plane_start..plane_start + spatial],
                );
            }
        }
    }
    SdTensor::new([n, out_channels, h, w], out)
}

fn conv2d_1x1_gemm_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: &[f32],
    workers: usize,
) -> Result<SdTensor> {
    let [n, in_channels, h, w] = shape4(input, "conv2d input")?;
    let [out_channels, _, _, _] = shape4(weight, "conv2d weight")?;
    let spatial = h * w;
    let input_batch_len = in_channels * spatial;
    let output_batch_len = out_channels * spatial;
    let mut out = vec![0.0; n * output_batch_len];
    for batch in 0..n {
        let out_batch = &mut out[batch * output_batch_len..(batch + 1) * output_batch_len];
        for (row, bias) in out_batch.chunks_mut(spatial).zip(bias.iter().copied()) {
            row.fill(bias);
        }
        let input_batch = &input.data[batch * input_batch_len..(batch + 1) * input_batch_len];
        unsafe {
            gemm::gemm(
                out_channels,
                spatial,
                in_channels,
                out_batch.as_mut_ptr(),
                1,
                spatial as isize,
                true,
                weight.data.as_ptr(),
                1,
                in_channels as isize,
                input_batch.as_ptr(),
                1,
                spatial as isize,
                1.0f32,
                1.0f32,
                false,
                false,
                false,
                Parallelism::Rayon(workers),
            );
        }
    }
    SdTensor::new([n, out_channels, h, w], out)
}

fn conv2d_3x3_pad1_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: &[f32],
    allow_parallel: bool,
) -> Result<SdTensor> {
    let [n, in_channels, h, w] = shape4(input, "conv2d input")?;
    let [out_channels, _, _, _] = shape4(weight, "conv2d weight")?;
    let spatial = h * w;
    let output_planes = n * out_channels;
    let mut out = vec![0.0; output_planes * spatial];
    let estimated_mul_adds = output_planes
        .saturating_mul(spatial)
        .saturating_mul(in_channels)
        .saturating_mul(9);
    let available_workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    const PARALLEL_CONV2D_THRESHOLD: usize = 1_000_000;

    if allow_parallel
        && available_workers > 1
        && out_channels >= 16
        && in_channels >= 16
        && estimated_mul_adds >= GEMM_CONV3X3_THRESHOLD
    {
        return conv2d_3x3_pad1_gemm_nchw(input, weight, bias, available_workers);
    }

    if allow_parallel
        && output_planes < available_workers
        && h >= 16
        && estimated_mul_adds >= PARALLEL_CONV2D_THRESHOLD
    {
        return conv2d_3x3_pad1_spatial_parallel_nchw(input, weight, bias, available_workers);
    }

    let workers = available_workers.min(output_planes);
    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_CONV2D_THRESHOLD {
        let planes_per_chunk = output_planes.div_ceil(workers);
        let values_per_chunk = planes_per_chunk * spatial;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let input_data = &input.data;
                let weight_data = &weight.data;
                let first_plane = chunk_index * planes_per_chunk;
                scope.spawn(move || {
                    for (local_plane, out_plane) in out_chunk.chunks_mut(spatial).enumerate() {
                        let plane = first_plane + local_plane;
                        let batch = plane / out_channels;
                        let out_channel = plane % out_channels;
                        fill_conv2d_3x3_pad1_plane(
                            input_data,
                            weight_data,
                            bias,
                            batch,
                            out_channel,
                            in_channels,
                            h,
                            w,
                            out_plane,
                        );
                    }
                });
            }
        });
    } else {
        for batch in 0..n {
            for out_channel in 0..out_channels {
                let plane_start = (batch * out_channels + out_channel) * spatial;
                fill_conv2d_3x3_pad1_plane(
                    &input.data,
                    &weight.data,
                    bias,
                    batch,
                    out_channel,
                    in_channels,
                    h,
                    w,
                    &mut out[plane_start..plane_start + spatial],
                );
            }
        }
    }
    SdTensor::new([n, out_channels, h, w], out)
}

fn conv2d_3x3_pad1_gemm_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: &[f32],
    workers: usize,
) -> Result<SdTensor> {
    let [n, in_channels, h, w] = shape4(input, "conv2d input")?;
    let [out_channels, _, _, _] = shape4(weight, "conv2d weight")?;
    let spatial = h * w;
    let output_planes = n * out_channels;
    let input_batch_len = in_channels * spatial;
    let output_batch_len = out_channels * spatial;
    let mut out = vec![0.0; output_planes * spatial];

    for batch in 0..n {
        let out_batch = &mut out[batch * output_batch_len..(batch + 1) * output_batch_len];
        for (row, bias) in out_batch.chunks_mut(spatial).zip(bias.iter().copied()) {
            row.fill(bias);
        }
        let input_batch = &input.data[batch * input_batch_len..(batch + 1) * input_batch_len];
        let full_width_taps = [
            (w, 0usize, (h - 1) * w, 1usize),
            (0usize, 0usize, spatial, 4usize),
            (0usize, w, (h - 1) * w, 7usize),
        ];
        for (out_offset, input_offset, cols, weight_offset) in full_width_taps {
            unsafe {
                gemm::gemm(
                    out_channels,
                    cols,
                    in_channels,
                    out_batch.as_mut_ptr().add(out_offset),
                    1,
                    spatial as isize,
                    true,
                    weight.data.as_ptr().add(weight_offset),
                    9,
                    (in_channels * 9) as isize,
                    input_batch.as_ptr().add(input_offset),
                    1,
                    spatial as isize,
                    1.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    Parallelism::Rayon(workers),
                );
            }
        }
        let side_taps = [
            (0usize, 0usize),
            (0usize, 2usize),
            (1usize, 0usize),
            (1usize, 2usize),
            (2usize, 0usize),
            (2usize, 2usize),
        ];
        if spatial <= PACKED_SIDE_TAP_MAX_SPATIAL {
            let packed_cols = h * (w - 1);
            let mut packed_input = vec![0.0; in_channels * packed_cols];
            let mut packed_out = vec![0.0; out_channels * packed_cols];
            for (kernel_y, kernel_x) in side_taps {
                fill_conv2d_3x3_side_tap_input(
                    input_batch,
                    in_channels,
                    h,
                    w,
                    kernel_y,
                    kernel_x,
                    &mut packed_input,
                );
                unsafe {
                    gemm::gemm(
                        out_channels,
                        packed_cols,
                        in_channels,
                        packed_out.as_mut_ptr(),
                        1,
                        packed_cols as isize,
                        false,
                        weight.data.as_ptr().add(kernel_y * 3 + kernel_x),
                        9,
                        (in_channels * 9) as isize,
                        packed_input.as_ptr(),
                        1,
                        packed_cols as isize,
                        0.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        Parallelism::Rayon(workers),
                    );
                }
                add_conv2d_3x3_side_tap_output(
                    &packed_out,
                    out_channels,
                    h,
                    w,
                    kernel_y,
                    kernel_x,
                    out_batch,
                );
            }
            continue;
        }
        for (kernel_y, kernel_x) in side_taps {
            let (out_y_start, out_y_end) = match kernel_y {
                0 => (1, h),
                1 => (0, h),
                _ => (0, h.saturating_sub(1)),
            };
            let (out_x_start, input_x_start) = match kernel_x {
                0 => (1, 0),
                _ => (0, 1),
            };
            let cols = w - 1;
            let weight_offset = kernel_y * 3 + kernel_x;
            for out_y in out_y_start..out_y_end {
                let input_y = out_y + kernel_y - 1;
                let out_offset = out_y * w + out_x_start;
                let input_offset = input_y * w + input_x_start;
                unsafe {
                    gemm::gemm(
                        out_channels,
                        cols,
                        in_channels,
                        out_batch.as_mut_ptr().add(out_offset),
                        1,
                        spatial as isize,
                        true,
                        weight.data.as_ptr().add(weight_offset),
                        9,
                        (in_channels * 9) as isize,
                        input_batch.as_ptr().add(input_offset),
                        1,
                        spatial as isize,
                        1.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        Parallelism::Rayon(workers),
                    );
                }
            }
        }
    }
    SdTensor::new([n, out_channels, h, w], out)
}

fn fill_conv2d_3x3_side_tap_input(
    input: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_y: usize,
    kernel_x: usize,
    out: &mut [f32],
) {
    let spatial = height * width;
    let packed_cols = height * (width - 1);
    let input_x_start = if kernel_x == 0 { 0 } else { 1 };
    for in_channel in 0..in_channels {
        let input_plane = &input[in_channel * spatial..(in_channel + 1) * spatial];
        let out_channel = &mut out[in_channel * packed_cols..(in_channel + 1) * packed_cols];
        for out_y in 0..height {
            let input_y = match kernel_y {
                0 => out_y.checked_sub(1),
                1 => Some(out_y),
                _ => (out_y + 1 < height).then_some(out_y + 1),
            };
            let out_start = out_y * (width - 1);
            if let Some(input_y) = input_y {
                let input_start = input_y * width + input_x_start;
                out_channel[out_start..out_start + width - 1]
                    .copy_from_slice(&input_plane[input_start..input_start + width - 1]);
            } else {
                out_channel[out_start..out_start + width - 1].fill(0.0);
            }
        }
    }
}

fn add_conv2d_3x3_side_tap_output(
    packed: &[f32],
    out_channels: usize,
    height: usize,
    width: usize,
    kernel_y: usize,
    kernel_x: usize,
    out: &mut [f32],
) {
    let spatial = height * width;
    let packed_cols = height * (width - 1);
    let out_x_start = if kernel_x == 0 { 1 } else { 0 };
    for out_channel in 0..out_channels {
        let packed_plane = &packed[out_channel * packed_cols..(out_channel + 1) * packed_cols];
        let out_plane = &mut out[out_channel * spatial..(out_channel + 1) * spatial];
        for out_y in 0..height {
            let valid_y = match kernel_y {
                0 => out_y > 0,
                1 => true,
                _ => out_y + 1 < height,
            };
            if valid_y {
                let packed_start = out_y * (width - 1);
                let out_start = out_y * width + out_x_start;
                for (dst, value) in out_plane[out_start..out_start + width - 1].iter_mut().zip(
                    packed_plane[packed_start..packed_start + width - 1]
                        .iter()
                        .copied(),
                ) {
                    *dst += value;
                }
            }
        }
    }
}

fn conv2d_3x3_pad1_spatial_parallel_nchw(
    input: &SdTensor,
    weight: &SdTensor,
    bias: &[f32],
    workers: usize,
) -> Result<SdTensor> {
    let [n, in_channels, h, w] = shape4(input, "conv2d input")?;
    let [out_channels, _, _, _] = shape4(weight, "conv2d weight")?;
    let spatial = h * w;
    let output_planes = n * out_channels;
    let mut out = vec![0.0; output_planes * spatial];
    let chunks_per_plane = workers.div_ceil(output_planes).clamp(1, h);
    let rows_per_chunk = h.div_ceil(chunks_per_plane);
    let values_per_chunk = rows_per_chunk * w;

    thread::scope(|scope| {
        for (plane, out_plane) in out.chunks_mut(spatial).enumerate() {
            let batch = plane / out_channels;
            let out_channel = plane % out_channels;
            for (chunk_index, out_rows) in out_plane.chunks_mut(values_per_chunk).enumerate() {
                let input_data = &input.data;
                let weight_data = &weight.data;
                let row_start = chunk_index * rows_per_chunk;
                scope.spawn(move || {
                    fill_conv2d_3x3_pad1_plane_rows(
                        input_data,
                        weight_data,
                        bias,
                        batch,
                        out_channel,
                        in_channels,
                        h,
                        w,
                        row_start,
                        out_rows,
                    );
                });
            }
        }
    });
    SdTensor::new([n, out_channels, h, w], out)
}

#[allow(clippy::too_many_arguments)]
fn fill_conv2d_1x1_plane(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    out_channel: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    out: &mut [f32],
) {
    let spatial = height * width;
    out.fill(bias[out_channel]);
    let input_batch_base = batch * in_channels * spatial;
    let weight_base = out_channel * in_channels;
    for in_channel in 0..in_channels {
        let scale = weight[weight_base + in_channel];
        let input_start = input_batch_base + in_channel * spatial;
        let input_plane = &input[input_start..input_start + spatial];
        for (dst, src) in out.iter_mut().zip(input_plane.iter().copied()) {
            *dst += src * scale;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_conv2d_3x3_pad1_plane_rows(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    out_channel: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    row_start: usize,
    out: &mut [f32],
) {
    let spatial = height * width;
    let row_end = row_start + out.len() / width;
    out.fill(bias[out_channel]);
    let input_batch_base = batch * in_channels * spatial;
    for in_channel in 0..in_channels {
        let input_start = input_batch_base + in_channel * spatial;
        let input_plane = &input[input_start..input_start + spatial];
        let weight_base = (out_channel * in_channels + in_channel) * 9;
        for kernel_y in 0..3 {
            let (valid_y_start, valid_y_end) = match kernel_y {
                0 => (1, height),
                1 => (0, height),
                _ => (0, height.saturating_sub(1)),
            };
            let out_y_start = valid_y_start.max(row_start);
            let out_y_end = valid_y_end.min(row_end);
            if out_y_start >= out_y_end {
                continue;
            }
            for kernel_x in 0..3 {
                let (out_x_start, out_x_end) = match kernel_x {
                    0 => (1, width),
                    1 => (0, width),
                    _ => (0, width.saturating_sub(1)),
                };
                if out_x_start >= out_x_end {
                    continue;
                }
                let scale = weight[weight_base + kernel_y * 3 + kernel_x];
                let values = out_x_end - out_x_start;
                for out_y in out_y_start..out_y_end {
                    let input_y = out_y + kernel_y - 1;
                    let input_x_start = out_x_start + kernel_x - 1;
                    let out_start = (out_y - row_start) * width + out_x_start;
                    let input_start = input_y * width + input_x_start;
                    add_scaled_slice(
                        &mut out[out_start..out_start + values],
                        &input_plane[input_start..input_start + values],
                        scale,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_conv2d_3x3_pad1_plane(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    out_channel: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    out: &mut [f32],
) {
    let spatial = height * width;
    out.fill(bias[out_channel]);
    let input_batch_base = batch * in_channels * spatial;
    for in_channel in 0..in_channels {
        let input_start = input_batch_base + in_channel * spatial;
        let input_plane = &input[input_start..input_start + spatial];
        let weight_base = (out_channel * in_channels + in_channel) * 9;
        for kernel_y in 0..3 {
            let (out_y_start, out_y_end) = match kernel_y {
                0 => (1, height),
                1 => (0, height),
                _ => (0, height.saturating_sub(1)),
            };
            if out_y_start >= out_y_end {
                continue;
            }
            for kernel_x in 0..3 {
                let (out_x_start, out_x_end) = match kernel_x {
                    0 => (1, width),
                    1 => (0, width),
                    _ => (0, width.saturating_sub(1)),
                };
                if out_x_start >= out_x_end {
                    continue;
                }
                let scale = weight[weight_base + kernel_y * 3 + kernel_x];
                let values = out_x_end - out_x_start;
                for out_y in out_y_start..out_y_end {
                    let input_y = out_y + kernel_y - 1;
                    let input_x_start = out_x_start + kernel_x - 1;
                    let out_start = out_y * width + out_x_start;
                    let input_start = input_y * width + input_x_start;
                    add_scaled_slice(
                        &mut out[out_start..out_start + values],
                        &input_plane[input_start..input_start + values],
                        scale,
                    );
                }
            }
        }
    }
}

fn add_scaled_slice(out: &mut [f32], input: &[f32], scale: f32) {
    for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
        *dst += src * scale;
    }
}

fn dot_slices(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| left * right)
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn fill_conv2d_plane(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    out_channel: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: Conv2dOptions,
    out: &mut [f32],
) {
    for out_y in 0..out_h {
        let y_origin = out_y * options.stride;
        let kernel_y_start = options.padding.saturating_sub(y_origin);
        let kernel_y_end = (options.padding + in_h)
            .saturating_sub(y_origin)
            .min(kernel_h);
        for out_x in 0..out_w {
            let x_origin = out_x * options.stride;
            let kernel_x_start = options.padding.saturating_sub(x_origin);
            let kernel_x_end = (options.padding + in_w)
                .saturating_sub(x_origin)
                .min(kernel_w);
            let mut sum = bias[out_channel];
            for in_channel in 0..in_channels {
                for kernel_y in kernel_y_start..kernel_y_end {
                    let in_y = y_origin + kernel_y - options.padding;
                    for kernel_x in kernel_x_start..kernel_x_end {
                        let in_x = x_origin + kernel_x - options.padding;
                        let input_index =
                            ((batch * in_channels + in_channel) * in_h + in_y) * in_w + in_x;
                        let weight_index = ((out_channel * in_channels + in_channel) * kernel_h
                            + kernel_y)
                            * kernel_w
                            + kernel_x;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
            out[out_y * out_w + out_x] = sum;
        }
    }
}

pub fn group_norm_nchw(
    input: &SdTensor,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<SdTensor> {
    group_norm_nchw_impl(input, groups, gamma, beta, eps, false, true)
}

pub fn group_norm_silu_nchw(
    input: &SdTensor,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<SdTensor> {
    group_norm_nchw_impl(input, groups, gamma, beta, eps, true, true)
}

fn group_norm_nchw_impl(
    input: &SdTensor,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    apply_silu: bool,
    allow_parallel: bool,
) -> Result<SdTensor> {
    let [n, c, h, w] = shape4(input, "group norm")?;
    if groups == 0 || c % groups != 0 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "group norm requires groups > 0 and channels divisible by groups, got c={c}, groups={groups}"
        )));
    }
    if gamma.len() != c || beta.len() != c {
        return Err(StableDiffusionError::InvalidInput(format!(
            "group norm expected {c} gamma/beta values, got {} and {}",
            gamma.len(),
            beta.len()
        )));
    }
    let channels_per_group = c / groups;
    let group_len = channels_per_group * h * w;
    let mut out = vec![0.0; input.data.len()];
    let group_count = n * groups;
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(group_count);

    if allow_parallel
        && workers > 1
        && input.data.len() >= PARALLEL_GROUP_NORM_THRESHOLD
        && group_count > 1
    {
        let groups_per_chunk = group_count.div_ceil(workers);
        let values_per_chunk = groups_per_chunk * group_len;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let input_data = &input.data;
                let first_group_index = chunk_index * groups_per_chunk;
                scope.spawn(move || {
                    for (local_group_index, out_group) in
                        out_chunk.chunks_mut(group_len).enumerate()
                    {
                        let group_index = first_group_index + local_group_index;
                        let batch = group_index / groups;
                        let group = group_index % groups;
                        fill_group_norm_group(
                            input_data,
                            gamma,
                            beta,
                            batch,
                            group,
                            c,
                            h,
                            w,
                            channels_per_group,
                            eps,
                            apply_silu,
                            out_group,
                        );
                    }
                });
            }
        });
    } else {
        for batch in 0..n {
            for group in 0..groups {
                let out_start = (batch * c + group * channels_per_group) * h * w;
                fill_group_norm_group(
                    &input.data,
                    gamma,
                    beta,
                    batch,
                    group,
                    c,
                    h,
                    w,
                    channels_per_group,
                    eps,
                    apply_silu,
                    &mut out[out_start..out_start + group_len],
                );
            }
        }
    }
    SdTensor::new(input.shape.clone(), out)
}

#[allow(clippy::too_many_arguments)]
fn fill_group_norm_group(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    group: usize,
    channels: usize,
    height: usize,
    width: usize,
    channels_per_group: usize,
    eps: f32,
    apply_silu: bool,
    out: &mut [f32],
) {
    let spatial = height * width;
    let channel_start = group * channels_per_group;
    let input_start = (batch * channels + channel_start) * spatial;
    let input_group = &input[input_start..input_start + out.len()];
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for value in input_group.iter().copied() {
        sum += value;
        square_sum += value * value;
    }
    let mean = sum / input_group.len() as f32;
    let variance = (square_sum / input_group.len() as f32 - mean * mean).max(0.0);
    let inv_std = 1.0 / (variance + eps).sqrt();

    for local_channel in 0..channels_per_group {
        let global_channel = channel_start + local_channel;
        let start = local_channel * spatial;
        let end = start + spatial;
        for (dst, src) in out[start..end]
            .iter_mut()
            .zip(input_group[start..end].iter().copied())
        {
            let value = (src - mean) * inv_std * gamma[global_channel] + beta[global_channel];
            *dst = if apply_silu {
                value / (1.0 + (-value).exp())
            } else {
                value
            };
        }
    }
}

pub fn layer_norm_last_dim(
    input: &SdTensor,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<SdTensor> {
    layer_norm_last_dim_impl(input, gamma, beta, eps, true)
}

fn layer_norm_last_dim_impl(
    input: &SdTensor,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    allow_parallel: bool,
) -> Result<SdTensor> {
    let cols = *input.shape.last().ok_or_else(|| {
        StableDiffusionError::InvalidInput("layer norm requires rank >= 1".to_string())
    })?;
    if gamma.len() != cols || beta.len() != cols {
        return Err(StableDiffusionError::InvalidInput(format!(
            "layer norm expected {cols} gamma/beta values, got {} and {}",
            gamma.len(),
            beta.len()
        )));
    }
    let rows = input.data.len() / cols;
    let mut out = vec![0.0; input.data.len()];
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(rows);

    if allow_parallel && workers > 1 && input.data.len() >= PARALLEL_LAYER_NORM_THRESHOLD {
        let rows_per_chunk = rows.div_ceil(workers);
        let values_per_chunk = rows_per_chunk * cols;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let input_data = &input.data;
                let first_row = chunk_index * rows_per_chunk;
                scope.spawn(move || {
                    for (local_row, out_row) in out_chunk.chunks_mut(cols).enumerate() {
                        let row = first_row + local_row;
                        let start = row * cols;
                        fill_layer_norm_row(
                            &input_data[start..start + cols],
                            gamma,
                            beta,
                            eps,
                            out_row,
                        );
                    }
                });
            }
        });
    } else {
        for row in 0..rows {
            let start = row * cols;
            fill_layer_norm_row(
                &input.data[start..start + cols],
                gamma,
                beta,
                eps,
                &mut out[start..start + cols],
            );
        }
    }
    SdTensor::new(input.shape.clone(), out)
}

fn fill_layer_norm_row(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for value in input.iter().copied() {
        sum += value;
        square_sum += value * value;
    }
    let mean = sum / input.len() as f32;
    let variance = (square_sum / input.len() as f32 - mean * mean).max(0.0);
    let inv_std = 1.0 / (variance + eps).sqrt();
    for col in 0..input.len() {
        out[col] = (input[col] - mean) * inv_std * gamma[col] + beta[col];
    }
}

pub fn matmul2d(left: &SdTensor, right: &SdTensor) -> Result<SdTensor> {
    matmul2d_impl(left, right, true)
}

pub fn linear2d(
    input: &SdTensor,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> Result<SdTensor> {
    linear2d_impl(input, weight, bias, in_features, out_features, true)
}

pub fn linear_flattened_last_dim(
    input: &SdTensor,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> Result<SdTensor> {
    if input.rank() < 2 || input.shape[input.rank() - 1] != in_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "linear_flattened_last_dim expected trailing dim {in_features}, got {:?}",
            input.shape
        )));
    }
    linear_from_data(
        &input.data,
        input.data.len() / in_features,
        weight,
        bias,
        in_features,
        out_features,
        true,
    )
}

fn linear2d_impl(
    input: &SdTensor,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    allow_parallel: bool,
) -> Result<SdTensor> {
    if input.rank() != 2 || input.shape[1] != in_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "linear2d expected input [rows, {in_features}], got {:?}",
            input.shape
        )));
    }
    if weight.len() != in_features * out_features {
        return Err(StableDiffusionError::InvalidInput(format!(
            "linear2d weight expected {} values, got {}",
            in_features * out_features,
            weight.len()
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != out_features {
            return Err(StableDiffusionError::InvalidInput(format!(
                "linear2d bias expected {out_features} values, got {}",
                bias.len()
            )));
        }
    }

    let rows = input.shape[0];
    linear_from_data(
        &input.data,
        rows,
        weight,
        bias,
        in_features,
        out_features,
        allow_parallel,
    )
}

fn linear_from_data(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    allow_parallel: bool,
) -> Result<SdTensor> {
    let mut out = vec![0.0; rows * out_features];
    let estimated_mul_adds = rows
        .saturating_mul(out_features)
        .saturating_mul(in_features);
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(rows);

    if allow_parallel && workers > 1 && estimated_mul_adds >= GEMM_LINEAR_THRESHOLD {
        fill_linear2d_gemm(
            input,
            weight,
            bias,
            rows,
            in_features,
            out_features,
            &mut out,
            workers,
        );
        return SdTensor::new([rows, out_features], out);
    }

    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_MATMUL_THRESHOLD {
        let rows_per_chunk = rows.div_ceil(workers);
        let values_per_chunk = rows_per_chunk * out_features;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let first_row = chunk_index * rows_per_chunk;
                scope.spawn(move || {
                    for (local_row, out_row) in out_chunk.chunks_mut(out_features).enumerate() {
                        let row = first_row + local_row;
                        fill_linear2d_row(
                            input,
                            weight,
                            bias,
                            row,
                            in_features,
                            out_features,
                            out_row,
                        );
                    }
                });
            }
        });
    } else {
        for row in 0..rows {
            let out_start = row * out_features;
            fill_linear2d_row(
                input,
                weight,
                bias,
                row,
                in_features,
                out_features,
                &mut out[out_start..out_start + out_features],
            );
        }
    }
    SdTensor::new([rows, out_features], out)
}

#[allow(clippy::too_many_arguments)]
fn fill_linear2d_gemm(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    in_features: usize,
    out_features: usize,
    out: &mut [f32],
    workers: usize,
) {
    unsafe {
        gemm::gemm(
            rows,
            out_features,
            in_features,
            out.as_mut_ptr(),
            1,
            out_features as isize,
            false,
            input.as_ptr(),
            1,
            in_features as isize,
            weight.as_ptr(),
            1,
            out_features as isize,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            Parallelism::Rayon(workers),
        );
    }
    if let Some(bias) = bias {
        for row in out.chunks_mut(out_features) {
            for (value, bias) in row.iter_mut().zip(bias.iter().copied()) {
                *value += bias;
            }
        }
    }
}

fn matmul2d_impl(left: &SdTensor, right: &SdTensor, allow_parallel: bool) -> Result<SdTensor> {
    if left.rank() != 2 || right.rank() != 2 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "matmul2d requires rank-2 tensors, got {:?} and {:?}",
            left.shape, right.shape
        )));
    }
    let rows = left.shape[0];
    let inner = left.shape[1];
    if right.shape[0] != inner {
        return Err(StableDiffusionError::InvalidInput(format!(
            "matmul2d inner dimensions do not match: {} vs {}",
            inner, right.shape[0]
        )));
    }
    let cols = right.shape[1];
    let mut out = vec![0.0; rows * cols];

    let estimated_mul_adds = rows.saturating_mul(cols).saturating_mul(inner);
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(rows);
    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_MATMUL_THRESHOLD {
        let rows_per_chunk = rows.div_ceil(workers);
        let values_per_chunk = rows_per_chunk * cols;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let left_data = &left.data;
                let right_data = &right.data;
                let first_row = chunk_index * rows_per_chunk;
                scope.spawn(move || {
                    for (local_row, out_row) in out_chunk.chunks_mut(cols).enumerate() {
                        let row = first_row + local_row;
                        fill_matmul2d_row(left_data, right_data, row, inner, cols, out_row);
                    }
                });
            }
        });
    } else {
        for row in 0..rows {
            let out_start = row * cols;
            fill_matmul2d_row(
                &left.data,
                &right.data,
                row,
                inner,
                cols,
                &mut out[out_start..out_start + cols],
            );
        }
    }
    SdTensor::new([rows, cols], out)
}

pub fn batched_matmul3d(left: &SdTensor, right: &SdTensor) -> Result<SdTensor> {
    batched_matmul3d_impl(left, right, true)
}

fn batched_matmul3d_impl(
    left: &SdTensor,
    right: &SdTensor,
    allow_parallel: bool,
) -> Result<SdTensor> {
    if left.rank() != 3 || right.rank() != 3 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "batched_matmul3d requires rank-3 tensors, got {:?} and {:?}",
            left.shape, right.shape
        )));
    }
    let [batch, rows, inner] = [left.shape[0], left.shape[1], left.shape[2]];
    if right.shape[0] != batch || right.shape[1] != inner {
        return Err(StableDiffusionError::InvalidInput(format!(
            "batched matmul shapes {:?} and {:?} are incompatible",
            left.shape, right.shape
        )));
    }
    let cols = right.shape[2];
    let mut out = vec![0.0; batch * rows * cols];

    let output_rows = batch * rows;
    let estimated_mul_adds = output_rows.saturating_mul(cols).saturating_mul(inner);
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(output_rows);

    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_MATMUL_THRESHOLD {
        let output_rows_per_chunk = output_rows.div_ceil(workers);
        let values_per_chunk = output_rows_per_chunk * cols;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let left_data = &left.data;
                let right_data = &right.data;
                let first_output_row = chunk_index * output_rows_per_chunk;
                scope.spawn(move || {
                    for (local_output_row, out_row) in out_chunk.chunks_mut(cols).enumerate() {
                        let output_row = first_output_row + local_output_row;
                        let b = output_row / rows;
                        let row = output_row % rows;
                        fill_batched_matmul3d_row(
                            left_data, right_data, b, row, rows, inner, cols, out_row,
                        );
                    }
                });
            }
        });
    } else {
        for b in 0..batch {
            for row in 0..rows {
                let out_start = (b * rows + row) * cols;
                fill_batched_matmul3d_row(
                    &left.data,
                    &right.data,
                    b,
                    row,
                    rows,
                    inner,
                    cols,
                    &mut out[out_start..out_start + cols],
                );
            }
        }
    }
    SdTensor::new([batch, rows, cols], out)
}

fn fill_matmul2d_row(
    left: &[f32],
    right: &[f32],
    row: usize,
    inner: usize,
    cols: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    let left_base = row * inner;
    for k in 0..inner {
        let left_value = left[left_base + k];
        let right_base = k * cols;
        for col in 0..cols {
            out[col] += left_value * right[right_base + col];
        }
    }
}

fn fill_linear2d_row(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    row: usize,
    in_features: usize,
    out_features: usize,
    out: &mut [f32],
) {
    match bias {
        Some(bias) => out.copy_from_slice(bias),
        None => out.fill(0.0),
    }
    let input_base = row * in_features;
    for k in 0..in_features {
        let input_value = input[input_base + k];
        let weight_base = k * out_features;
        for col in 0..out_features {
            out[col] += input_value * weight[weight_base + col];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_batched_matmul3d_row(
    left: &[f32],
    right: &[f32],
    batch: usize,
    row: usize,
    rows: usize,
    inner: usize,
    cols: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    let left_base = (batch * rows + row) * inner;
    let right_batch_base = batch * inner * cols;
    for k in 0..inner {
        let left_value = left[left_base + k];
        let right_base = right_batch_base + k * cols;
        for col in 0..cols {
            out[col] += left_value * right[right_base + col];
        }
    }
}

pub fn softmax_last_dim(input: &SdTensor) -> Result<SdTensor> {
    let cols = *input.shape.last().ok_or_else(|| {
        StableDiffusionError::InvalidInput("softmax requires rank >= 1".to_string())
    })?;
    let mut out = input.data.clone();
    for row in out.chunks_exact_mut(cols) {
        let max = row
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row {
            *value /= sum;
        }
    }
    SdTensor::new(input.shape.clone(), out)
}

pub fn scaled_dot_product_attention(
    query: &SdTensor,
    key: &SdTensor,
    value: &SdTensor,
    additive_mask: Option<&SdTensor>,
) -> Result<SdTensor> {
    scaled_dot_product_attention_impl(query, key, value, additive_mask, true)
}

fn scaled_dot_product_attention_impl(
    query: &SdTensor,
    key: &SdTensor,
    value: &SdTensor,
    additive_mask: Option<&SdTensor>,
    allow_parallel: bool,
) -> Result<SdTensor> {
    let [batch, heads, query_len, dim] = shape4(query, "attention query")?;
    let [key_batch, key_heads, key_len, key_dim] = shape4(key, "attention key")?;
    let [value_batch, value_heads, value_len, value_dim] = shape4(value, "attention value")?;
    if (key_batch, key_heads, key_dim) != (batch, heads, dim) {
        return Err(StableDiffusionError::InvalidInput(format!(
            "attention key shape {:?} is incompatible with query shape {:?}",
            key.shape, query.shape
        )));
    }
    if (value_batch, value_heads, value_len) != (batch, heads, key_len) {
        return Err(StableDiffusionError::InvalidInput(format!(
            "attention value shape {:?} is incompatible with key shape {:?}",
            value.shape, key.shape
        )));
    }
    if let Some(mask) = additive_mask {
        broadcast_shape(mask.shape(), &[batch, heads, query_len, key_len])?;
    }

    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0; batch * heads * query_len * value_dim];
    let query_rows = batch * heads * query_len;
    let estimated_mul_adds = query_rows
        .saturating_mul(key_len)
        .saturating_mul(dim + value_dim);
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(query_rows);

    if allow_parallel && workers > 1 && estimated_mul_adds >= PARALLEL_ATTENTION_THRESHOLD {
        let rows_per_chunk = query_rows.div_ceil(workers);
        let values_per_chunk = rows_per_chunk * value_dim;
        thread::scope(|scope| {
            for (chunk_index, out_chunk) in out.chunks_mut(values_per_chunk).enumerate() {
                let first_query_row = chunk_index * rows_per_chunk;
                let query_data = &query.data;
                let key_data = &key.data;
                let value_data = &value.data;
                scope.spawn(move || {
                    let mut scores = vec![0.0; key_len];
                    for (local_query_row, out_row) in out_chunk.chunks_mut(value_dim).enumerate() {
                        let query_row = first_query_row + local_query_row;
                        let b = query_row / (heads * query_len);
                        let within_batch = query_row % (heads * query_len);
                        let h = within_batch / query_len;
                        let q = within_batch % query_len;
                        fill_attention_query_row(
                            query_data,
                            key_data,
                            value_data,
                            additive_mask,
                            b,
                            h,
                            q,
                            batch,
                            heads,
                            query_len,
                            key_len,
                            dim,
                            value_dim,
                            scale,
                            &mut scores,
                            out_row,
                        );
                    }
                });
            }
        });
    } else {
        let mut scores = vec![0.0; key_len];
        for b in 0..batch {
            for h in 0..heads {
                for q in 0..query_len {
                    let out_index = nchw_index(b, h, q, 0, heads, query_len, value_dim);
                    fill_attention_query_row(
                        &query.data,
                        &key.data,
                        &value.data,
                        additive_mask,
                        b,
                        h,
                        q,
                        batch,
                        heads,
                        query_len,
                        key_len,
                        dim,
                        value_dim,
                        scale,
                        &mut scores,
                        &mut out[out_index..out_index + value_dim],
                    );
                }
            }
        }
    }
    SdTensor::new([batch, heads, query_len, value_dim], out)
}

#[allow(clippy::too_many_arguments)]
fn fill_attention_query_row(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    additive_mask: Option<&SdTensor>,
    batch_index: usize,
    head: usize,
    query_index: usize,
    batch: usize,
    heads: usize,
    query_len: usize,
    key_len: usize,
    dim: usize,
    value_dim: usize,
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
) {
    let query_base = ((batch_index * heads + head) * query_len + query_index) * dim;
    let query_row = &query[query_base..query_base + dim];
    let key_base = (batch_index * heads + head) * key_len * dim;
    for key_index in 0..key_len {
        let key_start = key_base + key_index * dim;
        let key_row = &key[key_start..key_start + dim];
        let dot = dot_slices(query_row, key_row);
        scores[key_index] = dot * scale;
        if let Some(mask) = additive_mask {
            let mask_indices = [batch_index, head, query_index, key_index];
            let mask_index = broadcast_linear_index(
                &mask_indices,
                &[batch, heads, query_len, key_len],
                mask.shape(),
            );
            scores[key_index] += mask.data[mask_index];
        }
    }
    softmax_slice_in_place(scores);
    out.fill(0.0);
    let value_base = (batch_index * heads + head) * key_len * value_dim;
    for (key_index, score) in scores.iter().copied().enumerate() {
        let value_start = value_base + key_index * value_dim;
        add_scaled_slice(out, &value[value_start..value_start + value_dim], score);
    }
}

fn tensor_len(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| StableDiffusionError::InvalidInput("tensor shape overflow".to_string()))
}

fn dim_from_right(shape: &[usize], offset: usize) -> Option<usize> {
    shape
        .len()
        .checked_sub(1 + offset)
        .map(|index| shape[index])
}

fn broadcast_linear_index(
    out_indices: &[usize],
    out_shape: &[usize],
    source_shape: &[usize],
) -> usize {
    let rank_offset = out_shape.len() - source_shape.len();
    let mut source_indices = vec![0; source_shape.len()];
    for (source_axis, source_dim) in source_shape.iter().copied().enumerate() {
        source_indices[source_axis] = if source_dim == 1 {
            0
        } else {
            out_indices[rank_offset + source_axis]
        };
    }
    indices_to_linear(&source_indices, source_shape)
}

fn shape4(tensor: &SdTensor, name: &str) -> Result<[usize; 4]> {
    if tensor.shape.len() != 4 {
        return Err(StableDiffusionError::InvalidInput(format!(
            "{name} requires rank-4 NCHW tensor, got {:?}",
            tensor.shape
        )));
    }
    Ok([
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    ])
}

fn nchw_index(
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    ((batch * channels + channel) * height + y) * width + x
}

fn linear_to_indices(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = index % shape[axis];
        index /= shape[axis];
    }
    indices
}

fn indices_to_linear(indices: &[usize], shape: &[usize]) -> usize {
    indices
        .iter()
        .zip(shape.iter())
        .fold(0usize, |acc, (index, dim)| acc * dim + index)
}

fn softmax_slice_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "actual {actual} expected {expected}"
            );
        }
    }

    fn conv2d_generic_reference(
        input: &SdTensor,
        weight: &SdTensor,
        bias: &[f32],
        options: Conv2dOptions,
    ) -> SdTensor {
        let [n, in_channels, in_h, in_w] = shape4(input, "conv2d input").unwrap();
        let [out_channels, _, kernel_h, kernel_w] = shape4(weight, "conv2d weight").unwrap();
        let padded_h = in_h + options.padding * 2;
        let padded_w = in_w + options.padding * 2;
        let out_h = (padded_h - kernel_h) / options.stride + 1;
        let out_w = (padded_w - kernel_w) / options.stride + 1;
        let plane_len = out_h * out_w;
        let mut out = vec![0.0; n * out_channels * plane_len];
        for batch in 0..n {
            for out_channel in 0..out_channels {
                let plane_start = (batch * out_channels + out_channel) * plane_len;
                fill_conv2d_plane(
                    input.data(),
                    weight.data(),
                    bias,
                    batch,
                    out_channel,
                    in_channels,
                    in_h,
                    in_w,
                    kernel_h,
                    kernel_w,
                    out_h,
                    out_w,
                    options,
                    &mut out[plane_start..plane_start + plane_len],
                );
            }
        }
        SdTensor::new([n, out_channels, out_h, out_w], out).unwrap()
    }

    fn broadcast_binary_generic_reference(
        left: &SdTensor,
        right: &SdTensor,
        op: impl Fn(f32, f32) -> f32,
    ) -> SdTensor {
        let shape = broadcast_shape(&left.shape, &right.shape).unwrap();
        let len = tensor_len(&shape).unwrap();
        let mut data = Vec::with_capacity(len);
        for index in 0..len {
            let out_indices = linear_to_indices(index, &shape);
            let left_index = broadcast_linear_index(&out_indices, &shape, &left.shape);
            let right_index = broadcast_linear_index(&out_indices, &shape, &right.shape);
            data.push(op(left.data[left_index], right.data[right_index]));
        }
        SdTensor::new(shape, data).unwrap()
    }

    fn concat_tensors_generic_reference(axis: usize, tensors: &[SdTensor]) -> SdTensor {
        let first = tensors.first().unwrap();
        let mut shape = first.shape.clone();
        shape[axis] = tensors.iter().map(|tensor| tensor.shape[axis]).sum();
        let mut data = vec![0.0; tensor_len(&shape).unwrap()];
        let mut axis_offset = 0;
        for tensor in tensors {
            for src_index in 0..tensor.data.len() {
                let mut out_indices = linear_to_indices(src_index, &tensor.shape);
                out_indices[axis] += axis_offset;
                let dst_index = indices_to_linear(&out_indices, &shape);
                data[dst_index] = tensor.data[src_index];
            }
            axis_offset += tensor.shape[axis];
        }
        SdTensor::new(shape, data).unwrap()
    }

    fn upsample_nearest2d_generic_reference(input: &SdTensor, scale: usize) -> SdTensor {
        let [n, c, h, w] = shape4(input, "nearest upsample").unwrap();
        let out_h = h * scale;
        let out_w = w * scale;
        let mut out = vec![0.0; n * c * out_h * out_w];
        for batch in 0..n {
            for channel in 0..c {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let src = nchw_index(batch, channel, y / scale, x / scale, c, h, w);
                        let dst = nchw_index(batch, channel, y, x, c, out_h, out_w);
                        out[dst] = input.data[src];
                    }
                }
            }
        }
        SdTensor::new([n, c, out_h, out_w], out).unwrap()
    }

    #[test]
    fn validates_shape_and_stats() {
        let tensor = SdTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(tensor.require_shape(&[2, 2]).is_ok());
        let stats = tensor.stats().unwrap();
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 4.0);
        assert_eq!(stats.mean, 2.5);
        assert!((stats.rms - 2.738613).abs() < 1e-5);
    }

    #[test]
    fn supports_elementwise_operations_and_broadcasting() {
        let left = SdTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let right = SdTensor::new([2], vec![10.0, 20.0]).unwrap();
        let same_shape = SdTensor::new([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let scalar = SdTensor::new([1], vec![2.0]).unwrap();

        assert_eq!(left.add(&right).unwrap().data(), &[11.0, 22.0, 13.0, 24.0]);
        assert_eq!(
            left.add(&same_shape).unwrap().data(),
            &[6.0, 8.0, 10.0, 12.0]
        );
        assert_eq!(left.mul(&scalar).unwrap().data(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(left.scale(2.0).unwrap().data(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(left.clamp(1.5, 3.5).unwrap().data(), &[1.5, 2.0, 3.0, 3.5]);
    }

    #[test]
    fn same_shape_binary_fast_path_matches_generic_reference() {
        let left = SdTensor::new(
            [2, 16, 32, 32],
            (0..2 * 16 * 32 * 32)
                .map(|index| ((index % 37) as f32 - 18.0) / 19.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [2, 16, 32, 32],
            (0..2 * 16 * 32 * 32)
                .map(|index| ((index % 29) as f32 - 14.0) / 17.0)
                .collect(),
        )
        .unwrap();

        let generic = broadcast_binary_generic_reference(&left, &right, |left, right| left + right);
        let fast = left.add(&right).unwrap();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-6);
    }

    #[test]
    #[ignore]
    fn same_shape_binary_fast_path_benchmark_smoke() {
        let left = SdTensor::new(
            [1, 320, 64, 64],
            (0..320 * 64 * 64)
                .map(|index| ((index % 37) as f32 - 18.0) / 19.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [1, 320, 64, 64],
            (0..320 * 64 * 64)
                .map(|index| ((index % 29) as f32 - 14.0) / 17.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let generic = broadcast_binary_generic_reference(&left, &right, |left, right| left + right);
        let generic_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fast = left.add(&right).unwrap();
        let fast_elapsed = started.elapsed();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-6);
        eprintln!(
            "same-shape binary benchmark: generic={:.3}s fast={:.3}s speedup={:.2}x",
            generic_elapsed.as_secs_f64(),
            fast_elapsed.as_secs_f64(),
            generic_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn concatenates_and_splits_nonzero_axis() {
        let left = SdTensor::new([1, 1, 2], vec![1.0, 2.0]).unwrap();
        let right = SdTensor::new([1, 2, 2], vec![3.0, 4.0, 5.0, 6.0]).unwrap();

        let concat = concat_tensors(1, &[left.clone(), right.clone()]).unwrap();

        assert_eq!(concat.shape(), &[1, 3, 2]);
        assert_eq!(concat.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let split = split_tensor(1, &concat, &[1, 2]).unwrap();
        assert_eq!(split, vec![left, right]);
    }

    #[test]
    fn concat_fast_path_matches_generic_reference() {
        let left = SdTensor::new(
            [2, 32, 16, 16],
            (0..2 * 32 * 16 * 16)
                .map(|index| ((index % 31) as f32 - 15.0) / 23.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [2, 48, 16, 16],
            (0..2 * 48 * 16 * 16)
                .map(|index| ((index % 29) as f32 - 14.0) / 19.0)
                .collect(),
        )
        .unwrap();

        let generic = concat_tensors_generic_reference(1, &[left.clone(), right.clone()]);
        let fast = concat_tensors(1, &[left, right]).unwrap();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-6);
    }

    #[test]
    #[ignore]
    fn concat_fast_path_benchmark_smoke() {
        let left = SdTensor::new(
            [1, 640, 32, 32],
            (0..640 * 32 * 32)
                .map(|index| ((index % 31) as f32 - 15.0) / 23.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [1, 640, 32, 32],
            (0..640 * 32 * 32)
                .map(|index| ((index % 29) as f32 - 14.0) / 19.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let generic = concat_tensors_generic_reference(1, &[left.clone(), right.clone()]);
        let generic_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fast = concat_tensors(1, &[left, right]).unwrap();
        let fast_elapsed = started.elapsed();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-6);
        eprintln!(
            "concat benchmark: generic={:.3}s fast={:.3}s speedup={:.2}x",
            generic_elapsed.as_secs_f64(),
            fast_elapsed.as_secs_f64(),
            generic_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn applies_channel_affine_and_nearest_resize() {
        let input = SdTensor::new([1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let affine = channel_affine_nchw(&input, &[2.0, 3.0], &[0.5, -1.0]).unwrap();
        assert_eq!(affine.data(), &[2.5, 4.5, 8.0, 11.0]);
        let up = upsample_nearest2d_nchw(&input, 2).unwrap();
        assert_eq!(up.shape(), &[1, 2, 2, 4]);
        assert_eq!(
            up.data(),
            &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
        );
        assert_eq!(downsample_nearest2d_nchw(&up, 2).unwrap(), input);
    }

    #[test]
    fn upsample_scale2_fast_path_matches_generic_reference() {
        let input = SdTensor::new(
            [2, 3, 5, 4],
            (0..2 * 3 * 5 * 4)
                .map(|index| ((index % 17) as f32 - 8.0) / 11.0)
                .collect(),
        )
        .unwrap();

        let generic = upsample_nearest2d_generic_reference(&input, 2);
        let fast = upsample_nearest2d_nchw(&input, 2).unwrap();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-6);
    }

    #[test]
    #[ignore]
    fn upsample_scale2_fast_path_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 64, 256, 256],
            (0..64 * 256 * 256)
                .map(|index| ((index % 37) as f32 - 18.0) / 19.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let generic = upsample_nearest2d_generic_reference(&input, 2);
        let generic_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fast = upsample_nearest2d_nchw(&input, 2).unwrap();
        let fast_elapsed = started.elapsed();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-6);
        eprintln!(
            "upsample scale2 benchmark: generic={:.3}s fast={:.3}s speedup={:.2}x",
            generic_elapsed.as_secs_f64(),
            fast_elapsed.as_secs_f64(),
            generic_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn conv2d_nchw_supports_bias_stride_and_padding() {
        let input =
            SdTensor::new([1, 1, 3, 3], (1..=9).map(|value| value as f32).collect()).unwrap();
        let weight = SdTensor::new([1, 1, 2, 2], vec![1.0, 0.0, 0.0, -1.0]).unwrap();

        let out = conv2d_nchw(
            &input,
            &weight,
            Some(&[0.5]),
            Conv2dOptions {
                stride: 1,
                padding: 0,
            },
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.data(), &[-3.5, -3.5, -3.5, -3.5]);
    }

    #[test]
    fn parallel_conv2d_matches_serial_reference() {
        let input = SdTensor::new(
            [1, 8, 16, 16],
            (0..1 * 8 * 16 * 16)
                .map(|index| ((index % 29) as f32 - 14.0) / 17.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [16, 8, 3, 3],
            (0..16 * 8 * 3 * 3)
                .map(|index| ((index % 19) as f32 - 9.0) / 23.0)
                .collect(),
        )
        .unwrap();
        let bias = (0..16)
            .map(|index| (index as f32 - 8.0) / 31.0)
            .collect::<Vec<_>>();
        let options = Conv2dOptions {
            stride: 1,
            padding: 1,
        };

        let serial = conv2d_nchw_impl(&input, &weight, Some(&bias), options, false).unwrap();
        let parallel = conv2d_nchw_impl(&input, &weight, Some(&bias), options, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
    }

    #[test]
    fn conv2d_1x1_fast_path_matches_generic_reference() {
        let input = SdTensor::new(
            [1, 8, 8, 8],
            (0..8 * 8 * 8)
                .map(|index| ((index % 31) as f32 - 15.0) / 19.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [12, 8, 1, 1],
            (0..12 * 8)
                .map(|index| ((index % 17) as f32 - 8.0) / 13.0)
                .collect(),
        )
        .unwrap();
        let bias = (0..12)
            .map(|index| (index as f32 - 6.0) / 23.0)
            .collect::<Vec<_>>();
        let options = Conv2dOptions {
            stride: 1,
            padding: 0,
        };

        let generic = conv2d_generic_reference(&input, &weight, &bias, options);
        let fast = conv2d_nchw(&input, &weight, Some(&bias), options).unwrap();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-5);
    }

    #[test]
    fn conv2d_1x1_gemm_matches_direct_fast_path() {
        let input = SdTensor::new(
            [1, 16, 17, 19],
            (0..16 * 17 * 19)
                .map(|index| ((index % 41) as f32 - 20.0) / 37.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [13, 16, 1, 1],
            (0..13 * 16)
                .map(|index| ((index % 29) as f32 - 14.0) / 23.0)
                .collect(),
        )
        .unwrap();
        let bias = (0..13)
            .map(|index| (index as f32 - 6.0) / 23.0)
            .collect::<Vec<_>>();

        let direct = conv2d_1x1_direct_nchw(&input, &weight, &bias, false).unwrap();
        let gemm = conv2d_1x1_gemm_nchw(&input, &weight, &bias, 2).unwrap();

        assert_eq!(gemm.shape(), direct.shape());
        assert_close(gemm.data(), direct.data(), 1e-4);
    }

    #[test]
    fn conv2d_3x3_pad1_fast_path_matches_generic_reference() {
        let input = SdTensor::new(
            [1, 5, 7, 6],
            (0..5 * 7 * 6)
                .map(|index| ((index % 31) as f32 - 15.0) / 19.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [9, 5, 3, 3],
            (0..9 * 5 * 3 * 3)
                .map(|index| ((index % 17) as f32 - 8.0) / 13.0)
                .collect(),
        )
        .unwrap();
        let bias = (0..9)
            .map(|index| (index as f32 - 4.0) / 23.0)
            .collect::<Vec<_>>();
        let options = Conv2dOptions {
            stride: 1,
            padding: 1,
        };

        let generic = conv2d_generic_reference(&input, &weight, &bias, options);
        let fast = conv2d_nchw(&input, &weight, Some(&bias), options).unwrap();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-5);
    }

    #[test]
    fn conv2d_3x3_pad1_spatial_parallel_matches_fast_path() {
        let input = SdTensor::new(
            [1, 4, 13, 11],
            (0..4 * 13 * 11)
                .map(|index| ((index % 31) as f32 - 15.0) / 19.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [3, 4, 3, 3],
            (0..3 * 4 * 3 * 3)
                .map(|index| ((index % 17) as f32 - 8.0) / 13.0)
                .collect(),
        )
        .unwrap();
        let bias = (0..3)
            .map(|index| (index as f32 - 1.0) / 23.0)
            .collect::<Vec<_>>();

        let fast = conv2d_3x3_pad1_nchw(&input, &weight, &bias, false).unwrap();
        let spatial = conv2d_3x3_pad1_spatial_parallel_nchw(&input, &weight, &bias, 4).unwrap();

        assert_eq!(spatial.shape(), fast.shape());
        assert_close(spatial.data(), fast.data(), 1e-5);
    }

    #[test]
    fn conv2d_3x3_pad1_gemm_matches_direct_fast_path() {
        let input = SdTensor::new(
            [1, 16, 17, 19],
            (0..16 * 17 * 19)
                .map(|index| ((index % 41) as f32 - 20.0) / 37.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [13, 16, 3, 3],
            (0..13 * 16 * 3 * 3)
                .map(|index| ((index % 29) as f32 - 14.0) / 23.0)
                .collect(),
        )
        .unwrap();
        let bias = (0..13)
            .map(|index| (index as f32 - 6.0) / 23.0)
            .collect::<Vec<_>>();

        let direct = conv2d_3x3_pad1_nchw(&input, &weight, &bias, false).unwrap();
        let gemm = conv2d_3x3_pad1_gemm_nchw(&input, &weight, &bias, 2).unwrap();

        assert_eq!(gemm.shape(), direct.shape());
        assert_close(gemm.data(), direct.data(), 1e-4);
    }

    #[test]
    #[ignore]
    fn conv2d_parallel_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 64, 64, 64],
            (0..1 * 64 * 64 * 64)
                .map(|index| ((index % 37) as f32 - 18.0) / 29.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [64, 64, 3, 3],
            (0..64 * 64 * 3 * 3)
                .map(|index| ((index % 23) as f32 - 11.0) / 31.0)
                .collect(),
        )
        .unwrap();
        let bias = vec![0.0; 64];
        let options = Conv2dOptions {
            stride: 1,
            padding: 1,
        };

        let started = std::time::Instant::now();
        let serial = conv2d_nchw_impl(&input, &weight, Some(&bias), options, false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = conv2d_nchw_impl(&input, &weight, Some(&bias), options, true).unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
        eprintln!(
            "conv2d benchmark: serial={:.3}s parallel={:.3}s speedup={:.2}x",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64(),
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore]
    fn conv2d_3x3_pad1_fast_path_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 64, 64, 64],
            (0..64 * 64 * 64)
                .map(|index| ((index % 37) as f32 - 18.0) / 29.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [64, 64, 3, 3],
            (0..64 * 64 * 3 * 3)
                .map(|index| ((index % 23) as f32 - 11.0) / 31.0)
                .collect(),
        )
        .unwrap();
        let bias = vec![0.0; 64];
        let options = Conv2dOptions {
            stride: 1,
            padding: 1,
        };

        let started = std::time::Instant::now();
        let generic = conv2d_generic_reference(&input, &weight, &bias, options);
        let generic_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fast = conv2d_nchw(&input, &weight, Some(&bias), options).unwrap();
        let fast_elapsed = started.elapsed();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-5);
        eprintln!(
            "conv2d 3x3 pad1 benchmark: generic={:.3}s fast={:.3}s speedup={:.2}x",
            generic_elapsed.as_secs_f64(),
            fast_elapsed.as_secs_f64(),
            generic_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore]
    fn conv2d_3x3_pad1_spatial_parallel_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 128, 128, 128],
            (0..128 * 128 * 128)
                .map(|index| ((index % 37) as f32 - 18.0) / 29.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [3, 128, 3, 3],
            (0..3 * 128 * 3 * 3)
                .map(|index| ((index % 23) as f32 - 11.0) / 31.0)
                .collect(),
        )
        .unwrap();
        let bias = vec![0.0; 3];

        let started = std::time::Instant::now();
        let current = conv2d_3x3_pad1_nchw(&input, &weight, &bias, false).unwrap();
        let current_elapsed = started.elapsed();

        let workers = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let started = std::time::Instant::now();
        let spatial =
            conv2d_3x3_pad1_spatial_parallel_nchw(&input, &weight, &bias, workers).unwrap();
        let spatial_elapsed = started.elapsed();

        assert_eq!(spatial.shape(), current.shape());
        assert_close(spatial.data(), current.data(), 1e-4);
        eprintln!(
            "conv2d 3x3 spatial parallel benchmark: current={:.3}s spatial={:.3}s speedup={:.2}x",
            current_elapsed.as_secs_f64(),
            spatial_elapsed.as_secs_f64(),
            current_elapsed.as_secs_f64() / spatial_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore]
    fn conv2d_1x1_fast_path_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 320, 64, 64],
            (0..320 * 64 * 64)
                .map(|index| ((index % 41) as f32 - 20.0) / 37.0)
                .collect(),
        )
        .unwrap();
        let weight = SdTensor::new(
            [320, 320, 1, 1],
            (0..320 * 320)
                .map(|index| ((index % 29) as f32 - 14.0) / 23.0)
                .collect(),
        )
        .unwrap();
        let bias = vec![0.0; 320];
        let options = Conv2dOptions {
            stride: 1,
            padding: 0,
        };

        let started = std::time::Instant::now();
        let generic = conv2d_generic_reference(&input, &weight, &bias, options);
        let generic_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fast = conv2d_nchw(&input, &weight, Some(&bias), options).unwrap();
        let fast_elapsed = started.elapsed();

        assert_eq!(fast.shape(), generic.shape());
        assert_close(fast.data(), generic.data(), 1e-5);
        eprintln!(
            "conv2d 1x1 benchmark: generic={:.3}s fast={:.3}s speedup={:.2}x",
            generic_elapsed.as_secs_f64(),
            fast_elapsed.as_secs_f64(),
            generic_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn group_norm_normalizes_per_group() {
        let input = SdTensor::new([1, 2, 1, 2], vec![1.0, 3.0, 10.0, 14.0]).unwrap();

        let out = group_norm_nchw(&input, 2, &[1.0, 1.0], &[0.0, 0.0], 1e-5).unwrap();

        assert_close(
            out.data(),
            &[-0.999_995, 0.999_995, -0.999_999, 0.999_999],
            1e-4,
        );
    }

    #[test]
    fn parallel_group_norm_matches_serial_reference() {
        let input = SdTensor::new(
            [1, 32, 32, 32],
            (0..32 * 32 * 32)
                .map(|index| ((index % 53) as f32 - 26.0) / 47.0)
                .collect(),
        )
        .unwrap();
        let gamma = (0..32)
            .map(|index| 0.75 + index as f32 / 127.0)
            .collect::<Vec<_>>();
        let beta = (0..32)
            .map(|index| (index as f32 - 16.0) / 101.0)
            .collect::<Vec<_>>();

        let serial = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, false, false).unwrap();
        let parallel = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, false, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
    }

    #[test]
    fn group_norm_silu_matches_separate_ops() {
        let input = SdTensor::new(
            [1, 8, 8, 7],
            (0..8 * 8 * 7)
                .map(|index| ((index % 53) as f32 - 26.0) / 47.0)
                .collect(),
        )
        .unwrap();
        let gamma = (0..8)
            .map(|index| 0.75 + index as f32 / 127.0)
            .collect::<Vec<_>>();
        let beta = (0..8)
            .map(|index| (index as f32 - 4.0) / 101.0)
            .collect::<Vec<_>>();

        let separate = group_norm_nchw(&input, 4, &gamma, &beta, 1e-5)
            .unwrap()
            .silu()
            .unwrap();
        let fused = group_norm_silu_nchw(&input, 4, &gamma, &beta, 1e-5).unwrap();

        assert_eq!(fused.shape(), separate.shape());
        assert_close(fused.data(), separate.data(), 1e-6);
    }

    #[test]
    #[ignore]
    fn group_norm_parallel_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 320, 64, 64],
            (0..320 * 64 * 64)
                .map(|index| ((index % 59) as f32 - 29.0) / 53.0)
                .collect(),
        )
        .unwrap();
        let gamma = (0..320)
            .map(|index| 0.8 + index as f32 / 4096.0)
            .collect::<Vec<_>>();
        let beta = (0..320)
            .map(|index| (index as f32 - 160.0) / 4096.0)
            .collect::<Vec<_>>();

        let started = std::time::Instant::now();
        let serial = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, false, false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, false, true).unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
        eprintln!(
            "group_norm benchmark: serial={:.3}s parallel={:.3}s speedup={:.2}x",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64(),
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore]
    fn group_norm_silu_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 128, 512, 512],
            (0..128 * 512 * 512)
                .map(|index| ((index % 59) as f32 - 29.0) / 53.0)
                .collect(),
        )
        .unwrap();
        let gamma = (0..128)
            .map(|index| 0.8 + index as f32 / 4096.0)
            .collect::<Vec<_>>();
        let beta = (0..128)
            .map(|index| (index as f32 - 64.0) / 4096.0)
            .collect::<Vec<_>>();

        let started = std::time::Instant::now();
        let separate = group_norm_nchw(&input, 32, &gamma, &beta, 1e-5)
            .unwrap()
            .silu()
            .unwrap();
        let separate_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fused = group_norm_silu_nchw(&input, 32, &gamma, &beta, 1e-5).unwrap();
        let fused_elapsed = started.elapsed();

        assert_eq!(fused.shape(), separate.shape());
        assert_close(fused.data(), separate.data(), 1e-5);
        eprintln!(
            "group_norm_silu benchmark: separate={:.3}s fused={:.3}s speedup={:.2}x",
            separate_elapsed.as_secs_f64(),
            fused_elapsed.as_secs_f64(),
            separate_elapsed.as_secs_f64() / fused_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn layer_norm_softmax_and_matmul_are_deterministic() {
        let input = SdTensor::new([2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();

        let norm = layer_norm_last_dim(&input, &[1.0, 1.0], &[0.0, 0.0], 1e-5).unwrap();
        assert_close(
            norm.data(),
            &[-0.999_995, 0.999_995, -0.999_995, 0.999_995],
            1e-4,
        );
        let softmax = softmax_last_dim(&input).unwrap();
        assert_close(
            softmax.data(),
            &[0.119_202_92, 0.880_797, 0.119_202_92, 0.880_797],
            1e-5,
        );
        let right = SdTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            matmul2d(&input, &right).unwrap().data(),
            &[10.0, 14.0, 14.0, 20.0]
        );
    }

    #[test]
    fn parallel_layer_norm_matches_serial_reference() {
        let input = SdTensor::new(
            [128, 320],
            (0..128 * 320)
                .map(|index| ((index % 61) as f32 - 30.0) / 57.0)
                .collect(),
        )
        .unwrap();
        let gamma = (0..320)
            .map(|index| 0.9 + index as f32 / 2048.0)
            .collect::<Vec<_>>();
        let beta = (0..320)
            .map(|index| (index as f32 - 160.0) / 2048.0)
            .collect::<Vec<_>>();

        let serial = layer_norm_last_dim_impl(&input, &gamma, &beta, 1e-5, false).unwrap();
        let parallel = layer_norm_last_dim_impl(&input, &gamma, &beta, 1e-5, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
    }

    #[test]
    #[ignore]
    fn layer_norm_parallel_benchmark_smoke() {
        let input = SdTensor::new(
            [4096, 320],
            (0..4096 * 320)
                .map(|index| ((index % 67) as f32 - 33.0) / 61.0)
                .collect(),
        )
        .unwrap();
        let gamma = (0..320)
            .map(|index| 0.9 + index as f32 / 4096.0)
            .collect::<Vec<_>>();
        let beta = (0..320)
            .map(|index| (index as f32 - 160.0) / 4096.0)
            .collect::<Vec<_>>();

        let started = std::time::Instant::now();
        let serial = layer_norm_last_dim_impl(&input, &gamma, &beta, 1e-5, false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = layer_norm_last_dim_impl(&input, &gamma, &beta, 1e-5, true).unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
        eprintln!(
            "layer_norm benchmark: serial={:.3}s parallel={:.3}s speedup={:.2}x",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64(),
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn parallel_matmul_matches_serial_reference() {
        let left = SdTensor::new(
            [64, 96],
            (0..64 * 96)
                .map(|index| ((index % 41) as f32 - 20.0) / 37.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [96, 80],
            (0..96 * 80)
                .map(|index| ((index % 31) as f32 - 15.0) / 29.0)
                .collect(),
        )
        .unwrap();

        let serial = matmul2d_impl(&left, &right, false).unwrap();
        let parallel = matmul2d_impl(&left, &right, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
    }

    #[test]
    #[ignore]
    fn matmul_parallel_benchmark_smoke() {
        let left = SdTensor::new(
            [4096, 320],
            (0..4096 * 320)
                .map(|index| ((index % 43) as f32 - 21.0) / 41.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [320, 320],
            (0..320 * 320)
                .map(|index| ((index % 37) as f32 - 18.0) / 31.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let serial = matmul2d_impl(&left, &right, false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = matmul2d_impl(&left, &right, true).unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-4);
        eprintln!(
            "matmul benchmark: serial={:.3}s parallel={:.3}s speedup={:.2}x",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64(),
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn linear2d_matches_matmul_plus_bias() {
        let input =
            SdTensor::new([3, 4], (0..12).map(|index| index as f32 / 7.0).collect()).unwrap();
        let weight = (0..20)
            .map(|index| ((index % 9) as f32 - 4.0) / 5.0)
            .collect::<Vec<_>>();
        let bias = (0..5)
            .map(|index| (index as f32 - 2.0) / 11.0)
            .collect::<Vec<_>>();
        let weight_tensor = SdTensor::new([4, 5], weight.clone()).unwrap();

        let mut expected = matmul2d(&input, &weight_tensor).unwrap();
        for row in expected.data_mut().chunks_exact_mut(5) {
            for (value, bias) in row.iter_mut().zip(bias.iter()) {
                *value += *bias;
            }
        }
        let actual = linear2d(&input, &weight, Some(&bias), 4, 5).unwrap();

        assert_eq!(actual.shape(), expected.shape());
        assert_close(actual.data(), expected.data(), 1e-6);
    }

    #[test]
    #[ignore]
    fn linear2d_avoids_weight_clone_benchmark_smoke() {
        let input = SdTensor::new(
            [4096, 320],
            (0..4096 * 320)
                .map(|index| ((index % 43) as f32 - 21.0) / 41.0)
                .collect(),
        )
        .unwrap();
        let weight = (0..320 * 1280)
            .map(|index| ((index % 37) as f32 - 18.0) / 31.0)
            .collect::<Vec<_>>();
        let bias = (0..1280)
            .map(|index| ((index % 17) as f32 - 8.0) / 23.0)
            .collect::<Vec<_>>();

        let started = std::time::Instant::now();
        let weight_tensor = SdTensor::new([320, 1280], weight.clone()).unwrap();
        let mut cloned = matmul2d(&input, &weight_tensor).unwrap();
        for row in cloned.data_mut().chunks_exact_mut(1280) {
            for (value, bias) in row.iter_mut().zip(bias.iter()) {
                *value += *bias;
            }
        }
        let cloned_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let direct = linear2d(&input, &weight, Some(&bias), 320, 1280).unwrap();
        let direct_elapsed = started.elapsed();

        assert_eq!(direct.shape(), cloned.shape());
        assert_close(direct.data(), cloned.data(), 1e-4);
        eprintln!(
            "linear2d benchmark: cloned={:.3}s direct={:.3}s speedup={:.2}x",
            cloned_elapsed.as_secs_f64(),
            direct_elapsed.as_secs_f64(),
            cloned_elapsed.as_secs_f64() / direct_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn batched_matmul_and_activations_work() {
        let left = SdTensor::new([1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let right = SdTensor::new([1, 2, 1], vec![10.0, 20.0]).unwrap();

        let out = batched_matmul3d(&left, &right).unwrap();

        assert_eq!(out.shape(), &[1, 2, 1]);
        assert_eq!(out.data(), &[50.0, 110.0]);
        let silu = SdTensor::new([1], vec![0.0]).unwrap().silu().unwrap();
        assert_eq!(silu.data(), &[0.0]);
        let gelu = SdTensor::new([1], vec![0.0]).unwrap().gelu().unwrap();
        assert_eq!(gelu.data(), &[0.0]);
    }

    #[test]
    fn parallel_activations_match_serial_reference() {
        let input = SdTensor::new(
            [1, 32, 32, 32],
            (0..32 * 32 * 32)
                .map(|index| ((index % 43) as f32 - 21.0) / 17.0)
                .collect(),
        )
        .unwrap();

        let silu_serial = input.silu_impl(false).unwrap();
        let silu_parallel = input.silu_impl(true).unwrap();
        let gelu_serial = input.gelu_impl(false).unwrap();
        let gelu_parallel = input.gelu_impl(true).unwrap();

        assert_eq!(silu_parallel.shape(), silu_serial.shape());
        assert_eq!(gelu_parallel.shape(), gelu_serial.shape());
        assert_close(silu_parallel.data(), silu_serial.data(), 1e-6);
        assert_close(gelu_parallel.data(), gelu_serial.data(), 1e-6);
    }

    #[test]
    #[ignore]
    fn silu_parallel_benchmark_smoke() {
        let input = SdTensor::new(
            [1, 320, 64, 64],
            (0..320 * 64 * 64)
                .map(|index| ((index % 47) as f32 - 23.0) / 19.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let serial = input.silu_impl(false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = input.silu_impl(true).unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-6);
        eprintln!(
            "silu benchmark: serial={:.3}s parallel={:.3}s speedup={:.2}x",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64(),
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    fn parallel_batched_matmul_matches_serial_reference() {
        let left = SdTensor::new(
            [4, 32, 48],
            (0..4 * 32 * 48)
                .map(|index| ((index % 47) as f32 - 23.0) / 43.0)
                .collect(),
        )
        .unwrap();
        let right = SdTensor::new(
            [4, 48, 40],
            (0..4 * 48 * 40)
                .map(|index| ((index % 29) as f32 - 14.0) / 23.0)
                .collect(),
        )
        .unwrap();

        let serial = batched_matmul3d_impl(&left, &right, false).unwrap();
        let parallel = batched_matmul3d_impl(&left, &right, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
    }

    #[test]
    fn scaled_dot_product_attention_supports_cross_attention_and_mask() {
        let query = SdTensor::new([1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let key = SdTensor::new([1, 1, 3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let value = SdTensor::new([1, 1, 3, 1], vec![10.0, 20.0, 30.0]).unwrap();
        let mask = SdTensor::new([1, 1, 1, 3], vec![0.0, 0.0, -10_000.0]).unwrap();

        let out = scaled_dot_product_attention(&query, &key, &value, Some(&mask)).unwrap();

        assert_eq!(out.shape(), &[1, 1, 2, 1]);
        assert_close(out.data(), &[13.302_38, 16.697_62], 1e-4);
    }

    #[test]
    fn parallel_attention_matches_serial_reference() {
        let query = SdTensor::new(
            [1, 4, 128, 16],
            (0..4 * 128 * 16)
                .map(|index| ((index % 71) as f32 - 35.0) / 67.0)
                .collect(),
        )
        .unwrap();
        let key = SdTensor::new(
            [1, 4, 77, 16],
            (0..4 * 77 * 16)
                .map(|index| ((index % 61) as f32 - 30.0) / 59.0)
                .collect(),
        )
        .unwrap();
        let value = SdTensor::new(
            [1, 4, 77, 16],
            (0..4 * 77 * 16)
                .map(|index| ((index % 53) as f32 - 26.0) / 47.0)
                .collect(),
        )
        .unwrap();

        let serial = scaled_dot_product_attention_impl(&query, &key, &value, None, false).unwrap();
        let parallel = scaled_dot_product_attention_impl(&query, &key, &value, None, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
    }

    #[test]
    #[ignore]
    fn attention_score_reuse_benchmark_smoke() {
        let query = SdTensor::new(
            [1, 4, 512, 40],
            (0..4 * 512 * 40)
                .map(|index| ((index % 73) as f32 - 36.0) / 67.0)
                .collect(),
        )
        .unwrap();
        let key = SdTensor::new(
            [1, 4, 512, 40],
            (0..4 * 512 * 40)
                .map(|index| ((index % 67) as f32 - 33.0) / 61.0)
                .collect(),
        )
        .unwrap();
        let value = SdTensor::new(
            [1, 4, 512, 40],
            (0..4 * 512 * 40)
                .map(|index| ((index % 59) as f32 - 29.0) / 53.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let allocating = attention_allocating_reference(&query, &key, &value);
        let allocating_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let reused = scaled_dot_product_attention_impl(&query, &key, &value, None, false).unwrap();
        let reused_elapsed = started.elapsed();

        assert_eq!(reused.shape(), allocating.shape());
        assert_close(reused.data(), allocating.data(), 1e-5);
        eprintln!(
            "attention score reuse benchmark: allocating={:.3}s reused={:.3}s speedup={:.2}x",
            allocating_elapsed.as_secs_f64(),
            reused_elapsed.as_secs_f64(),
            allocating_elapsed.as_secs_f64() / reused_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore]
    fn attention_parallel_benchmark_smoke() {
        let query = SdTensor::new(
            [1, 8, 4096, 40],
            (0..8 * 4096 * 40)
                .map(|index| ((index % 73) as f32 - 36.0) / 67.0)
                .collect(),
        )
        .unwrap();
        let key = SdTensor::new(
            [1, 8, 77, 40],
            (0..8 * 77 * 40)
                .map(|index| ((index % 67) as f32 - 33.0) / 61.0)
                .collect(),
        )
        .unwrap();
        let value = SdTensor::new(
            [1, 8, 77, 40],
            (0..8 * 77 * 40)
                .map(|index| ((index % 59) as f32 - 29.0) / 53.0)
                .collect(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let serial = scaled_dot_product_attention_impl(&query, &key, &value, None, false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = scaled_dot_product_attention_impl(&query, &key, &value, None, true).unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
        eprintln!(
            "attention benchmark: serial={:.3}s parallel={:.3}s speedup={:.2}x",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64(),
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }

    fn attention_allocating_reference(
        query: &SdTensor,
        key: &SdTensor,
        value: &SdTensor,
    ) -> SdTensor {
        let [batch, heads, query_len, dim] = shape4(query, "attention query").unwrap();
        let [_, _, key_len, _] = shape4(key, "attention key").unwrap();
        let [_, _, _, value_dim] = shape4(value, "attention value").unwrap();
        let scale = 1.0 / (dim as f32).sqrt();
        let mut out = vec![0.0; batch * heads * query_len * value_dim];
        for b in 0..batch {
            for h in 0..heads {
                for q in 0..query_len {
                    let mut scores = vec![0.0; key_len];
                    for key_index in 0..key_len {
                        let mut dot = 0.0;
                        for d in 0..dim {
                            dot += query.data[nchw_index(b, h, q, d, heads, query_len, dim)]
                                * key.data[nchw_index(b, h, key_index, d, heads, key_len, dim)];
                        }
                        scores[key_index] = dot * scale;
                    }
                    softmax_slice_in_place(&mut scores);
                    for value_channel in 0..value_dim {
                        let mut sum = 0.0;
                        for (key_index, score) in scores.iter().copied().enumerate() {
                            sum += score
                                * value.data[nchw_index(
                                    b,
                                    h,
                                    key_index,
                                    value_channel,
                                    heads,
                                    key_len,
                                    value_dim,
                                )];
                        }
                        out[nchw_index(b, h, q, value_channel, heads, query_len, value_dim)] = sum;
                    }
                }
            }
        }
        SdTensor::new([batch, heads, query_len, value_dim], out).unwrap()
    }
}
