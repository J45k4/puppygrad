#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseShape {
    pub rows: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl DenseShape {
    pub fn new(rows: usize, in_features: usize, out_features: usize) -> Self {
        Self {
            rows,
            in_features,
            out_features,
        }
    }

    pub fn out_len(self) -> usize {
        self.rows * self.out_features
    }

    pub fn work_items(self) -> usize {
        self.rows * self.in_features * self.out_features
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedRows {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl QuantizedRows {
    pub fn from_f32(values: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut quantized = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(rows);
        for r in 0..rows {
            let src = row(values, r, cols);
            let max_abs = src.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales.push(scale);
            for value in src {
                let q = (value / scale).round().clamp(-127.0, 127.0) as i8;
                quantized.push(q);
            }
        }
        Self {
            values: quantized,
            scales,
            rows,
            cols,
        }
    }
}

pub fn transposed_dense_projection_into(
    x: &[f32],
    shape: DenseShape,
    weight: &[f32],
    bias: &[f32],
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(shape.out_len(), 0.0);
    for r in 0..shape.rows {
        let src = row(x, r, shape.in_features);
        let dst = row_mut(out, r, shape.out_features);
        for o in 0..shape.out_features {
            let weight_row = row(weight, o, shape.in_features);
            dst[o] = bias[o] + dot(src, weight_row);
        }
    }
}

pub fn dense_projection_into(
    x: &[f32],
    shape: DenseShape,
    weight: &[f32],
    bias: &[f32],
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(shape.out_len(), 0.0);
    for r in 0..shape.rows {
        let src = row(x, r, shape.in_features);
        let dst = row_mut(out, r, shape.out_features);
        for o in 0..shape.out_features {
            let mut sum = bias[o];
            for i in 0..shape.in_features {
                sum += src[i] * weight[i * shape.out_features + o];
            }
            dst[o] = sum;
        }
    }
}

pub fn quantized_transposed_dense_projection_into(
    x: &[f32],
    shape: DenseShape,
    weight: &QuantizedRows,
    bias: &[f32],
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(weight.rows, shape.out_features);
    debug_assert_eq!(weight.cols, shape.in_features);
    out.clear();
    out.resize(shape.out_len(), 0.0);
    for r in 0..shape.rows {
        let src = row(x, r, shape.in_features);
        let dst = row_mut(out, r, shape.out_features);
        for o in 0..shape.out_features {
            dst[o] = bias[o] + quantized_dot(src, weight, o);
        }
    }
}

pub fn layer_norm_in_place(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    for r in 0..rows {
        let row = row_mut(x, r, cols);
        let mean = row.iter().sum::<f32>() / cols as f32;
        let variance = row
            .iter()
            .map(|v| {
                let delta = *v - mean;
                delta * delta
            })
            .sum::<f32>()
            / cols as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for c in 0..cols {
            row[c] = (row[c] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
}

pub fn gelu_in_place(values: &mut [f32]) {
    for value in values {
        let x = *value;
        *value = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
    }
}

pub fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

pub fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

pub fn quantized_dot(a: &[f32], rows: &QuantizedRows, row_index: usize) -> f32 {
    debug_assert_eq!(a.len(), rows.cols);
    debug_assert!(row_index < rows.rows);
    let start = row_index * rows.cols;
    let values = &rows.values[start..start + rows.cols];
    let scale = rows.scales[row_index];
    let mut sum = 0.0f32;
    for (x, q) in a.iter().zip(values.iter()) {
        sum += *x * (*q as f32 * scale);
    }
    sum
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(a, b) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { dot_avx(a, b) };
        }
        dot_scalar(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

#[allow(dead_code)]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vld1q_f32, vmlaq_f32};

    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= a.len() {
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        acc = vmlaq_f32(acc, av, bv);
        i += 4;
    }

    let mut sum = vaddvq_f32(acc);
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f32>();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f32>();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

pub fn transpose_in_out(weight: &[f32], in_features: usize, out_features: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for i in 0..in_features {
        for o in 0..out_features {
            transposed[o * in_features + i] = weight[i * out_features + o];
        }
    }
    transposed
}

pub fn row(values: &[f32], row: usize, cols: usize) -> &[f32] {
    &values[row * cols..(row + 1) * cols]
}

pub fn row_mut(values: &mut [f32], row: usize, cols: usize) -> &mut [f32] {
    &mut values[row * cols..(row + 1) * cols]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_reference() {
        let a = patterned(37, 0.013);
        let b = patterned(37, -0.021);

        assert!((dot(&a, &b) - dot_scalar(&a, &b)).abs() <= 1e-6);
    }

    #[test]
    fn transposed_dense_projection_matches_dense_projection() {
        let x = patterned(6, 0.2);
        let weight = patterned(12, 0.1);
        let bias = vec![0.5, -0.25, 0.125, 0.0];
        let transposed = transpose_in_out(&weight, 3, 4);
        let shape = DenseShape::new(2, 3, 4);
        let mut dense = Vec::new();
        let mut transposed_dense = Vec::new();

        dense_projection_into(&x, shape, &weight, &bias, &mut dense);
        transposed_dense_projection_into(&x, shape, &transposed, &bias, &mut transposed_dense);

        assert_close(&dense, &transposed_dense, 1e-6);
    }

    #[test]
    fn quantized_projection_returns_finite_values() {
        let x = patterned(6, 0.2);
        let weight = patterned(12, 0.1);
        let bias = vec![0.5, -0.25, 0.125, 0.0];
        let rows = QuantizedRows::from_f32(&transpose_in_out(&weight, 3, 4), 4, 3);
        let mut out = Vec::new();

        quantized_transposed_dense_projection_into(
            &x,
            DenseShape::new(2, 3, 4),
            &rows,
            &bias,
            &mut out,
        );

        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|value| value.is_finite()));
    }

    fn patterned(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + 11) % 23) as f32 - 11.0) * scale)
            .collect()
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (*l - *r).abs() <= tolerance,
                "values differ at {i}: left={l} right={r}"
            );
        }
    }
}
