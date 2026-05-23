use crate::models::cpu;

use super::{BarkError, Result};

pub fn embedding_lookup(
    token_ids: &[usize],
    embedding: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if vocab_size == 0 || hidden_size == 0 {
        return Err(BarkError::InvalidInput(
            "embedding vocab and hidden sizes must be > 0".to_string(),
        ));
    }
    if embedding.len() != vocab_size * hidden_size {
        return Err(BarkError::InvalidInput(format!(
            "embedding length {} does not match vocab_size {vocab_size} x hidden_size {hidden_size}",
            embedding.len()
        )));
    }
    let mut out = Vec::with_capacity(token_ids.len() * hidden_size);
    for &token_id in token_ids {
        if token_id >= vocab_size {
            return Err(BarkError::InvalidInput(format!(
                "token id {token_id} is outside vocab size {vocab_size}"
            )));
        }
        let start = token_id * hidden_size;
        out.extend_from_slice(&embedding[start..start + hidden_size]);
    }
    Ok(out)
}

pub fn add_in_place(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(BarkError::InvalidInput(format!(
            "residual lengths differ: {} vs {}",
            dst.len(),
            src.len()
        )));
    }
    cpu::add_in_place(dst, src);
    Ok(())
}

pub fn layer_norm_in_place(
    values: &mut [f32],
    rows: usize,
    hidden_size: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<()> {
    if values.len() != rows * hidden_size {
        return Err(BarkError::InvalidInput(format!(
            "layer norm values length {} does not match rows {rows} x hidden_size {hidden_size}",
            values.len()
        )));
    }
    if weight.len() != hidden_size || bias.is_some_and(|bias| bias.len() != hidden_size) {
        return Err(BarkError::InvalidInput(
            "layer norm weight/bias shape does not match hidden size".to_string(),
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(BarkError::InvalidInput(
            "layer norm eps must be finite and > 0".to_string(),
        ));
    }
    let zero_bias;
    let bias = match bias {
        Some(bias) => bias,
        None => {
            zero_bias = vec![0.0; hidden_size];
            &zero_bias
        }
    };
    cpu::layer_norm_in_place(values, rows, hidden_size, weight, bias, eps);
    Ok(())
}

pub fn linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
) -> Result<Vec<f32>> {
    if input.len() != rows * in_features {
        return Err(BarkError::InvalidInput(format!(
            "linear input length {} does not match rows {rows} x in_features {in_features}",
            input.len()
        )));
    }
    if weight.len() != out_features * in_features {
        return Err(BarkError::InvalidInput(format!(
            "linear weight length {} does not match out_features {out_features} x in_features {in_features}",
            weight.len()
        )));
    }
    if bias.is_some_and(|bias| bias.len() != out_features) {
        return Err(BarkError::InvalidInput(
            "linear bias length does not match out features".to_string(),
        ));
    }
    let zero_bias;
    let bias = match bias {
        Some(bias) => bias,
        None => {
            zero_bias = vec![0.0; out_features];
            &zero_bias
        }
    };
    let mut out = Vec::new();
    cpu::transposed_dense_projection_into(
        input,
        cpu::DenseShape::new(rows, in_features, out_features),
        weight,
        bias,
        &mut out,
    );
    Ok(out)
}

pub fn gelu_in_place(values: &mut [f32]) {
    cpu::gelu_in_place(values);
}

pub fn causal_self_attention(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    causal_self_attention_with_mask(
        input,
        seq_len,
        hidden_size,
        num_heads,
        qkv_weight,
        qkv_bias,
        out_weight,
        out_bias,
        None,
    )
}

pub fn causal_self_attention_with_mask(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
    attention_mask: Option<&[bool]>,
) -> Result<Vec<f32>> {
    self_attention(
        input,
        seq_len,
        hidden_size,
        num_heads,
        qkv_weight,
        qkv_bias,
        out_weight,
        out_bias,
        true,
        attention_mask,
    )
}

pub fn full_self_attention(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    full_self_attention_with_mask(
        input,
        seq_len,
        hidden_size,
        num_heads,
        qkv_weight,
        qkv_bias,
        out_weight,
        out_bias,
        None,
    )
}

pub fn full_self_attention_with_mask(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
    attention_mask: Option<&[bool]>,
) -> Result<Vec<f32>> {
    self_attention(
        input,
        seq_len,
        hidden_size,
        num_heads,
        qkv_weight,
        qkv_bias,
        out_weight,
        out_bias,
        false,
        attention_mask,
    )
}

fn self_attention(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: Option<&[f32]>,
    out_weight: &[f32],
    out_bias: Option<&[f32]>,
    causal: bool,
    attention_mask: Option<&[bool]>,
) -> Result<Vec<f32>> {
    if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
        return Err(BarkError::InvalidInput(
            "hidden size must be divisible by num_heads".to_string(),
        ));
    }
    if attention_mask.is_some_and(|mask| mask.len() != seq_len) {
        return Err(BarkError::InvalidInput(format!(
            "attention mask length does not match seq_len {seq_len}"
        )));
    }
    let qkv = linear(
        input,
        seq_len,
        hidden_size,
        qkv_weight,
        qkv_bias,
        hidden_size * 3,
    )?;
    let head_dim = hidden_size / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut context = vec![0.0; seq_len * hidden_size];
    let mut scores = vec![0.0; seq_len];
    for pos in 0..seq_len {
        let attention_len = if causal { pos + 1 } else { seq_len };
        for head in 0..num_heads {
            let q_base = pos * hidden_size * 3 + head * head_dim;
            for key_pos in 0..attention_len {
                let k_base = key_pos * hidden_size * 3 + hidden_size + head * head_dim;
                let mut score = 0.0;
                for dim in 0..head_dim {
                    score += qkv[q_base + dim] * qkv[k_base + dim];
                }
                scores[key_pos] = if attention_mask.is_some_and(|mask| !mask[key_pos]) {
                    f32::NEG_INFINITY
                } else {
                    score * scale
                };
            }
            let has_attention = softmax_prefix_in_place(&mut scores, attention_len);
            if !has_attention {
                continue;
            }
            for dim in 0..head_dim {
                let mut value = 0.0;
                for key_pos in 0..attention_len {
                    let v_base = key_pos * hidden_size * 3 + hidden_size * 2 + head * head_dim;
                    value += scores[key_pos] * qkv[v_base + dim];
                }
                context[pos * hidden_size + head * head_dim + dim] = value;
            }
        }
    }
    linear(
        &context,
        seq_len,
        hidden_size,
        out_weight,
        out_bias,
        hidden_size,
    )
}

fn softmax_prefix_in_place(values: &mut [f32], len: usize) -> bool {
    let max = values[..len]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        for value in &mut values[..len] {
            *value = 0.0;
        }
        return false;
    }
    let mut total = 0.0;
    for value in &mut values[..len] {
        *value = (*value - max).exp();
        total += *value;
    }
    if total > 0.0 && total.is_finite() {
        for value in &mut values[..len] {
            *value /= total;
        }
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_lookup_extracts_rows() -> Result<()> {
        let embedding = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let out = embedding_lookup(&[2, 0], &embedding, 3, 2)?;

        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn embedding_lookup_rejects_bad_shapes_and_token_ids() {
        let shape_err = embedding_lookup(&[0], &[0.0, 1.0, 2.0], 2, 2).unwrap_err();
        assert!(shape_err.to_string().contains("embedding length"));

        let id_err = embedding_lookup(&[2], &[0.0, 1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(id_err.to_string().contains("outside vocab size"));
    }

    #[test]
    fn layer_norm_normalizes_rows() -> Result<()> {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];

        layer_norm_in_place(&mut values, 2, 2, &[1.0, 1.0], Some(&[0.0, 0.0]), 1e-5)?;

        assert!((values[0] + 0.999_98).abs() < 0.0001);
        assert!((values[1] - 0.999_98).abs() < 0.0001);
        assert!((values[2] + 0.999_98).abs() < 0.0001);
        assert!((values[3] - 0.999_98).abs() < 0.0001);
        Ok(())
    }

    #[test]
    fn layer_norm_rejects_bad_shapes_and_eps() {
        let mut bad_values = vec![1.0, 2.0, 3.0];
        let values_err =
            layer_norm_in_place(&mut bad_values, 2, 2, &[1.0, 1.0], None, 1e-5).unwrap_err();
        assert!(values_err.to_string().contains("values length"));

        let mut values = vec![1.0, 2.0];
        let weight_err = layer_norm_in_place(&mut values, 1, 2, &[1.0], None, 1e-5).unwrap_err();
        assert!(weight_err.to_string().contains("weight/bias shape"));

        let eps_err = layer_norm_in_place(&mut values, 1, 2, &[1.0, 1.0], None, 0.0).unwrap_err();
        assert!(eps_err.to_string().contains("eps"));
    }

    #[test]
    fn linear_applies_row_major_pytorch_weight() -> Result<()> {
        let out = linear(
            &[1.0, 2.0],
            1,
            2,
            &[3.0, 4.0, 5.0, 6.0],
            Some(&[0.5, -0.5]),
            2,
        )?;

        assert_eq!(out, vec![11.5, 16.5]);
        Ok(())
    }

    #[test]
    fn linear_rejects_bad_shapes() {
        let input_err = linear(&[1.0], 1, 2, &[1.0, 2.0], None, 1).unwrap_err();
        assert!(input_err.to_string().contains("linear input length"));

        let weight_err = linear(&[1.0, 2.0], 1, 2, &[1.0], None, 1).unwrap_err();
        assert!(weight_err.to_string().contains("linear weight length"));

        let bias_err = linear(&[1.0, 2.0], 1, 2, &[1.0, 2.0], Some(&[0.0, 0.0]), 1).unwrap_err();
        assert!(bias_err.to_string().contains("linear bias length"));
    }

    #[test]
    fn causal_attention_preserves_shape() -> Result<()> {
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let qkv_weight = vec![
            1.0, 0.0, 0.0, 1.0, // q
            1.0, 0.0, 0.0, 1.0, // k
            1.0, 0.0, 0.0, 1.0, // v
        ];
        let out_weight = vec![1.0, 0.0, 0.0, 1.0];

        let out = causal_self_attention(&input, 2, 2, 1, &qkv_weight, None, &out_weight, None)?;

        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 0.0001);
        assert!((out[1] - 0.0).abs() < 0.0001);
        assert!(out.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn full_attention_can_attend_to_future_tokens() -> Result<()> {
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let qkv_weight = vec![
            1.0, 0.0, 0.0, 1.0, // q
            1.0, 0.0, 0.0, 1.0, // k
            1.0, 0.0, 0.0, 1.0, // v
        ];
        let out_weight = vec![1.0, 0.0, 0.0, 1.0];

        let causal = causal_self_attention(&input, 2, 2, 1, &qkv_weight, None, &out_weight, None)?;
        let full = full_self_attention(&input, 2, 2, 1, &qkv_weight, None, &out_weight, None)?;

        assert_eq!(full.len(), 4);
        assert_ne!(full[0], causal[0]);
        assert!(full.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn attention_mask_blocks_invalid_key_positions() -> Result<()> {
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let qkv_weight = vec![
            1.0, 0.0, 0.0, 1.0, // q
            1.0, 0.0, 0.0, 1.0, // k
            1.0, 0.0, 0.0, 1.0, // v
        ];
        let out_weight = vec![1.0, 0.0, 0.0, 1.0];

        let out = full_self_attention_with_mask(
            &input,
            2,
            2,
            1,
            &qkv_weight,
            None,
            &out_weight,
            None,
            Some(&[true, false]),
        )?;

        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 0.0001);
        assert!((out[1] - 0.0).abs() < 0.0001);
        assert!((out[2] - 1.0).abs() < 0.0001);
        assert!((out[3] - 0.0).abs() < 0.0001);
        Ok(())
    }

    #[test]
    fn attention_rejects_invalid_head_and_mask_shapes() {
        let qkv_weight = vec![0.0; 3 * 3 * 3];
        let out_weight = vec![0.0; 3 * 3];
        let head_err = causal_self_attention(
            &[0.0, 0.0, 0.0],
            1,
            3,
            2,
            &qkv_weight,
            None,
            &out_weight,
            None,
        )
        .unwrap_err();
        assert!(head_err.to_string().contains("divisible by num_heads"));

        let mask_err = full_self_attention_with_mask(
            &[0.0, 0.0],
            1,
            2,
            1,
            &[0.0; 12],
            None,
            &[0.0; 4],
            None,
            Some(&[true, false]),
        )
        .unwrap_err();
        assert!(mask_err.to_string().contains("attention mask length"));
    }
}
