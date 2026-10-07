//! Preserve logical tensor coordinates when a retained allocation grows.
use super::pop::{numel, Error, Result};

pub(super) fn grow(
    bytes: &[u8],
    old: &[usize],
    new: &[usize],
    item_bytes: usize,
) -> Result<Vec<u8>> {
    if item_bytes == 0 {
        return Err(Error("state item width must be positive".into()));
    }
    if old.len() != new.len() || old.iter().zip(new).any(|(a, b)| a > b) {
        return Err(Error(
            "retained state growth requires matching ranks and nondecreasing dimensions".into(),
        ));
    }
    let old_bytes = numel(old)?
        .checked_mul(item_bytes)
        .ok_or_else(|| Error("state size overflow".into()))?;
    let new_bytes = numel(new)?
        .checked_mul(item_bytes)
        .ok_or_else(|| Error("state size overflow".into()))?;
    if bytes.len() != old_bytes || new_bytes > 2 * 1024 * 1024 * 1024 {
        return Err(Error("invalid or oversized retained state growth".into()));
    }
    let mut result = vec![0; new_bytes];
    if old_bytes == 0 {
        return Ok(result);
    }
    if old.len() <= 1 || old[1..] == new[1..] {
        result[..old_bytes].copy_from_slice(bytes);
        return Ok(result);
    }
    let width = old.last().copied().unwrap_or(1);
    let new_width = new.last().copied().unwrap_or(1);
    let row_bytes = width * item_bytes;
    for row in 0..old_bytes / row_bytes {
        let (mut index, mut offset, mut stride) = (row, 0, new_width);
        for axis in (0..old.len().saturating_sub(1)).rev() {
            offset += (index % old[axis]) * stride;
            index /= old[axis];
            stride *= new[axis];
        }
        let target = offset * item_bytes;
        result[target..target + row_bytes]
            .copy_from_slice(&bytes[row * row_bytes..(row + 1) * row_bytes]);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn growth_preserves_rows_across_inner_and_outer_dimensions() {
        assert_eq!(
            grow(&[1, 2, 3, 4, 5, 6], &[2, 3], &[3, 5], 1).unwrap(),
            [1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            grow(&[1, 2, 3, 4], &[2, 1, 2], &[2, 2, 3], 1).unwrap(),
            [1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0]
        );
        assert_eq!(
            grow(&[1, 2, 3, 4, 5, 6, 7, 8], &[2, 1], &[2, 2], 4).unwrap(),
            [1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0]
        );
        assert_eq!(grow(&[7], &[], &[], 1).unwrap(), [7]);
        assert_eq!(grow(&[], &[0, 2], &[1, 2], 1).unwrap(), [0, 0]);
        assert!(grow(&[], &[], &[], 0).is_err());
        assert!(grow(&[1, 2], &[2], &[1], 1).is_err());
    }
}
