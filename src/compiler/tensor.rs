//! Library decompositions, not additional compiler opcodes.

use super::pop::{numel, Arg, DType, Error, Graph, Op, ReduceOp, Result, Value};

pub fn input(graph: &mut Graph, slot: usize, dtype: DType, shape: &[usize]) -> Result<Value> {
    let param = graph.param(slot, dtype, Some(numel(shape)?))?;
    graph.reshape(param, shape)
}

pub fn matmul(graph: &mut Graph, a: Value, b: Value) -> Result<Value> {
    let a_shape = graph
        .node(a)?
        .shape()
        .ok_or_else(|| Error("matmul requires tensor values".into()))?;
    let b_shape = graph
        .node(b)?
        .shape()
        .ok_or_else(|| Error("matmul requires tensor values".into()))?;
    if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
        return Err(Error("matmul requires [M,K] and [K,N]".into()));
    }
    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
    let left = graph.reshape(a, &[m, 1, k])?;
    let right = graph.apply(Op::Permute, &[b], Arg::Axes(vec![1, 0]))?;
    let right = graph.reshape(right, &[1, n, k])?;
    let product = graph.apply(Op::Mul, &[left, right], Arg::None)?;
    let product = graph.apply(Op::Permute, &[product], Arg::Axes(vec![2, 0, 1]))?;
    graph.apply(
        Op::Reduce,
        &[product],
        Arg::Reduce {
            op: ReduceOp::Add,
            num_axes: 1,
        },
    )
}
