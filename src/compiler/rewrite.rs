//! Single-pass bottom-up rewriting. No hidden fixed-point iteration.

use super::pop::{Error, Graph, Op, Result, Value};
use std::collections::HashMap;

pub fn rewrite(
    graph: &mut Graph,
    root: Value,
    mut rule: impl FnMut(&mut Graph, Value) -> Result<Option<Value>>,
) -> Result<Value> {
    let mut rewritten = HashMap::new();
    for old in graph.toposort(root)? {
        let node = graph.node(old)?;
        let (op, arg) = (node.op(), node.arg().clone());
        let src = node.src().iter().map(|v| rewritten[v]).collect::<Vec<_>>();
        let rebuilt = graph.apply(op, &src, arg)?;
        let replacement = rule(graph, rebuilt)?.unwrap_or(rebuilt);
        let expected = graph.node(old)?;
        let actual = graph.node(replacement)?;
        if expected.dtype() != actual.dtype() || expected.shape() != actual.shape() {
            return Err(Error("rewrite must preserve dtype and shape".into()));
        }
        rewritten.insert(old, replacement);
    }
    Ok(rewritten[&root])
}

/// Identity views/casts only: x*0 and x+0 are not safe IEEE float rewrites.
pub fn simplify_views(graph: &mut Graph, root: Value) -> Result<Value> {
    rewrite(graph, root, |graph, value| {
        let node = graph.node(value)?;
        let Some(&input) = node.src().first() else {
            return Ok(None);
        };
        let source = graph.node(input)?;
        let identity = match node.op() {
            Op::Reshape | Op::Expand => node.shape() == source.shape(),
            Op::Cast => node.dtype() == source.dtype(),
            Op::Permute => match node.arg() {
                super::pop::Arg::Axes(axes) => axes.iter().copied().eq(0..axes.len()),
                _ => false,
            },
            _ => false,
        };
        if identity {
            return Ok(Some(input));
        }
        Ok(None)
    })
}
