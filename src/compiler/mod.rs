//! Compiler foundation: pure tensor graphs, library decompositions, and rewrites.
//!
//! Inspired by tinygrad's UOp architecture. See `docs/compiler-ir.md` for the
//! supported subset and the boundary between tensor values and execution.

pub mod autodiff;
pub mod cpu;
pub mod cuda;
mod expression;
pub mod pop;
pub mod rewrite;
pub mod source;
pub mod spec;
pub mod tensor;

#[cfg(test)]
mod tests;
