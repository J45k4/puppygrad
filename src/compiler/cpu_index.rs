//! Coordinate propagation through reshapes. Keep independent dimension groups
//! separate so spatial coordinates do not get mixed into reduction coordinates.
use super::{numel, strides, Result};

pub(super) fn reshape_coordinates(
    old: &[usize],
    new: &[usize],
    coords: &[String],
) -> Result<Vec<String>> {
    let mut result = vec!["0".into(); old.len()];
    if numel(old)? == 0 {
        return Ok(result);
    }
    let a = old
        .iter()
        .enumerate()
        .filter(|(_, d)| **d != 1)
        .collect::<Vec<_>>();
    let b = new
        .iter()
        .enumerate()
        .filter(|(_, d)| **d != 1)
        .collect::<Vec<_>>();
    let (mut i, mut j) = (0, 0);
    while i < a.len() {
        let (ai, bj) = (i, j);
        let (mut ap, mut bp) = (*a[i].1, *b[j].1);
        i += 1;
        j += 1;
        while ap != bp {
            if ap < bp {
                ap *= *a[i].1;
                i += 1;
            } else {
                bp *= *b[j].1;
                j += 1;
            }
        }
        let dims = b[bj..j].iter().map(|(_, d)| **d).collect::<Vec<_>>();
        let terms = b[bj..j]
            .iter()
            .zip(strides(&dims))
            .map(|((axis, _), stride)| {
                if stride == 1 {
                    format!("({})", coords[*axis])
                } else {
                    format!("({})*{stride}", coords[*axis])
                }
            })
            .collect::<Vec<_>>();
        let flat = terms.join("+");
        let dims = a[ai..i].iter().map(|(_, d)| **d).collect::<Vec<_>>();
        for ((axis, d), stride) in a[ai..i].iter().zip(strides(&dims)) {
            result[*axis] = if i - ai == 1 {
                flat.clone()
            } else if stride == 1 {
                format!("({flat})%{d}")
            } else if *axis == a[ai].0 {
                format!("({flat})/{stride}")
            } else {
                format!("(({flat})/{stride})%{d}")
            };
        }
    }
    Ok(result)
}

/// Propagate physical strides through only the reshape groups that remain
/// affine. Returning None affects the packing heuristic, never tensor semantics.
pub(super) fn reshape_strides(
    old: &[usize],
    new: &[usize],
    layout: &[isize],
) -> Option<Vec<isize>> {
    if old.contains(&0) || new.contains(&0) {
        return None;
    }
    let a = old
        .iter()
        .enumerate()
        .filter(|(_, d)| **d != 1)
        .collect::<Vec<_>>();
    let b = new
        .iter()
        .enumerate()
        .filter(|(_, d)| **d != 1)
        .collect::<Vec<_>>();
    let mut result = vec![0; new.len()];
    let (mut i, mut j) = (0, 0);
    while i < a.len() {
        let (ai, bj) = (i, j);
        let (mut ap, mut bp) = (*a[i].1, *b[j].1);
        i += 1;
        j += 1;
        while ap != bp {
            if ap < bp {
                ap *= *a[i].1;
                i += 1;
            } else {
                bp *= *b[j].1;
                j += 1;
            }
        }
        for pair in a[ai..i].windows(2) {
            if layout[pair[0].0] != layout[pair[1].0] * (*pair[1].1 as isize) {
                return None;
            }
        }
        let dims = b[bj..j].iter().map(|(_, d)| **d).collect::<Vec<_>>();
        for ((axis, _), stride) in b[bj..j].iter().zip(strides(&dims)) {
            result[*axis] = layout[a[i - 1].0] * stride as isize;
        }
    }
    Some(result)
}
