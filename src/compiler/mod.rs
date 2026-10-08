//! Compiler foundation: pure tensor graphs, library decompositions, and rewrites.
//!
//! Inspired by tinygrad's UOp architecture. See `docs/compiler-ir.md` for the
//! supported subset and the boundary between tensor values and execution.

pub mod autodiff;
pub mod cpu;
pub mod cuda;
pub mod device;
mod expression;
pub mod gpu;
pub mod hip;
mod kernel_cache;
pub mod pop;
pub mod rewrite;
pub mod source;
pub mod spec;
mod state_resize;
pub mod tensor;

#[cfg(test)]
mod tests;
