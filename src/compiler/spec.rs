//! Stage boundaries from tinygrad/uop/spec.py, for our implemented subset.

use super::pop::{DType, Error, Graph, Op, Result, Value};

pub const TINYGRAD_REVISION: &str = "1a58c3ae9d5ff5605d81085cf1a364c113e95cf6";

#[derive(Clone, Copy, Debug)]
pub enum Spec {
    Tensor,
    Program,
    KernelGraph,
}

/// Construction already checks supported tensor nodes. Later stages impose
/// stricter rules; accepting a stage does not mean code generation exists yet.
pub fn verify(graph: &Graph, root: Value, spec: Spec) -> Result<()> {
    for (index, value) in graph.toposort(root)?.into_iter().enumerate() {
        let node = graph.node(value)?;
        let allowed = match spec {
            Spec::Tensor => true,
            Spec::Program => {
                !matches!(
                    node.op(),
                    Op::Reshape
                        | Op::Expand
                        | Op::Permute
                        | Op::Pad
                        | Op::Shrink
                        | Op::Flip
                        | Op::Window
                        | Op::Reduce
                ) && (node.op() == Op::Const || !node.dtype().is_weak())
                    && (node.op() == Op::Cast
                        || node
                            .src()
                            .iter()
                            .all(|&v| graph.node(v).unwrap().op() != Op::Const))
            }
            Spec::KernelGraph => match node.op() {
                Op::Sink | Op::Param | Op::Const => true,
                Op::Cast => graph.node(node.src()[0])?.op() == Op::Const,
                Op::Stack => node
                    .src()
                    .iter()
                    .all(|&v| matches!(graph.node(v).unwrap().op(), Op::Const | Op::Param)),
                _ => false,
            },
        };
        if !allowed {
            return Err(Error(format!(
                "{spec:?} spec rejects node %{index}: {:?} {:?}",
                node.op(),
                node.dtype()
            )));
        }
        if matches!(spec, Spec::Program) && node.op() == Op::Cast && node.dtype() == DType::Void {
            return Err(Error("program CAST cannot produce void".into()));
        }
    }
    Ok(())
}
