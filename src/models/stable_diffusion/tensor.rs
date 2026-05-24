use std::thread;

use super::{Result, StableDiffusionError};

const PARALLEL_MATMUL_THRESHOLD: usize = 1_000_000;
const PARALLEL_GROUP_NORM_THRESHOLD: usize = 64 * 1024;
const PARALLEL_LAYER_NORM_THRESHOLD: usize = 64 * 1024;
const PARALLEL_ATTENTION_THRESHOLD: usize = 1_000_000;
const PARALLEL_ELEMENTWISE_THRESHOLD: usize = 256 * 1024;

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
    for spatial_index in 0..spatial {
        let mut sum = bias[out_channel];
        for in_channel in 0..in_channels {
            let input_index = (batch * in_channels + in_channel) * spatial + spatial_index;
            let weight_index = out_channel * in_channels + in_channel;
            sum += input[input_index] * weight[weight_index];
        }
        out[spatial_index] = sum;
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
    for y in 0..height {
        for x in 0..width {
            if y > 0 && y + 1 < height && x > 0 && x + 1 < width {
                let mut sum = bias[out_channel];
                for in_channel in 0..in_channels {
                    let input_base = ((batch * in_channels + in_channel) * height + y) * width + x;
                    let weight_base = (out_channel * in_channels + in_channel) * 9;
                    sum += input[input_base - width - 1] * weight[weight_base]
                        + input[input_base - width] * weight[weight_base + 1]
                        + input[input_base - width + 1] * weight[weight_base + 2]
                        + input[input_base - 1] * weight[weight_base + 3]
                        + input[input_base] * weight[weight_base + 4]
                        + input[input_base + 1] * weight[weight_base + 5]
                        + input[input_base + width - 1] * weight[weight_base + 6]
                        + input[input_base + width] * weight[weight_base + 7]
                        + input[input_base + width + 1] * weight[weight_base + 8];
                }
                out[y * width + x] = sum;
            } else {
                let mut sum = bias[out_channel];
                for in_channel in 0..in_channels {
                    for kernel_y in 0..3 {
                        let raw_y = y + kernel_y;
                        if raw_y == 0 || raw_y > height {
                            continue;
                        }
                        let in_y = raw_y - 1;
                        for kernel_x in 0..3 {
                            let raw_x = x + kernel_x;
                            if raw_x == 0 || raw_x > width {
                                continue;
                            }
                            let in_x = raw_x - 1;
                            let input_index =
                                ((batch * in_channels + in_channel) * height + in_y) * width + in_x;
                            let weight_index = (out_channel * in_channels + in_channel) * 9
                                + kernel_y * 3
                                + kernel_x;
                            sum += input[input_index] * weight[weight_index];
                        }
                    }
                }
                out[y * width + x] = sum;
            }
        }
    }
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
    group_norm_nchw_impl(input, groups, gamma, beta, eps, true)
}

fn group_norm_nchw_impl(
    input: &SdTensor,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
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
    out: &mut [f32],
) {
    let spatial = height * width;
    let channel_start = group * channels_per_group;
    let input_start = (batch * channels + channel_start) * spatial;
    let input_group = &input[input_start..input_start + out.len()];
    let mean = input_group.iter().sum::<f32>() / input_group.len() as f32;
    let variance = input_group
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / input_group.len() as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    for local_channel in 0..channels_per_group {
        let global_channel = channel_start + local_channel;
        let start = local_channel * spatial;
        let end = start + spatial;
        for (dst, src) in out[start..end]
            .iter_mut()
            .zip(input_group[start..end].iter().copied())
        {
            *dst = (src - mean) * inv_std * gamma[global_channel] + beta[global_channel];
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
    let mean = input.iter().sum::<f32>() / input.len() as f32;
    let variance = input
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / input.len() as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();
    for col in 0..input.len() {
        out[col] = (input[col] - mean) * inv_std * gamma[col] + beta[col];
    }
}

pub fn matmul2d(left: &SdTensor, right: &SdTensor) -> Result<SdTensor> {
    matmul2d_impl(left, right, true)
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
    for col in 0..cols {
        let mut sum = 0.0;
        for k in 0..inner {
            sum += left[row * inner + k] * right[k * cols + col];
        }
        out[col] = sum;
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
    for col in 0..cols {
        let mut sum = 0.0;
        for k in 0..inner {
            sum += left[(batch * rows + row) * inner + k] * right[(batch * inner + k) * cols + col];
        }
        out[col] = sum;
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
    for key_index in 0..key_len {
        let mut dot = 0.0;
        for d in 0..dim {
            dot += query[nchw_index(batch_index, head, query_index, d, heads, query_len, dim)]
                * key[nchw_index(batch_index, head, key_index, d, heads, key_len, dim)];
        }
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
    for value_channel in 0..value_dim {
        let mut sum = 0.0;
        for (key_index, score) in scores.iter().copied().enumerate() {
            sum += score
                * value[nchw_index(
                    batch_index,
                    head,
                    key_index,
                    value_channel,
                    heads,
                    key_len,
                    value_dim,
                )];
        }
        out[value_channel] = sum;
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

        let serial = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, false).unwrap();
        let parallel = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, true).unwrap();

        assert_eq!(parallel.shape(), serial.shape());
        assert_close(parallel.data(), serial.data(), 1e-5);
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
        let serial = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, false).unwrap();
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let parallel = group_norm_nchw_impl(&input, 32, &gamma, &beta, 1e-5, true).unwrap();
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
