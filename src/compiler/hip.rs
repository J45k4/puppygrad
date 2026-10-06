//! HIP kernels and execution through the shared GPU compiler/runtime.
use super::{
    gpu,
    pop::{Graph, Result, Value},
};
pub use gpu::{
    DeviceMemory, Executable, ExecutionStats, MemoryPlan, ParameterMemory, ResidencyStats,
};
use std::path::Path;

#[derive(Clone)]
pub struct Runtime(pub(crate) gpu::Runtime);
impl Runtime {
    pub fn new(device: usize) -> Result<Self> {
        Ok(Self(gpu::Runtime::new(gpu::Backend::Hip, device)?))
    }
}
impl std::ops::Deref for Runtime {
    type Target = gpu::Runtime;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
pub fn device_index(device: &str) -> Result<usize> {
    let (backend, index) = gpu::device(device)?;
    if backend != gpu::Backend::Hip {
        return Err(super::pop::Error(format!(
            "invalid HIP device {device:?}; expected hip:<index>"
        )));
    }
    Ok(index)
}
pub fn emit(g: &Graph, root: Value) -> Result<(String, usize)> {
    let x = super::cpu::cuda_lower::emit_backend(g, root, gpu::Backend::Hip)?;
    Ok((x.source, x.gemm_count))
}
pub fn memory_plan(g: &Graph, root: Value) -> Result<MemoryPlan> {
    gpu::memory_plan_for_backend(g, root, gpu::Backend::Hip)
}
pub fn compile(g: &Graph, root: Value, cache: &Path, device: usize) -> Result<Executable> {
    compile_with_runtime(g, root, cache, &Runtime::new(device)?)
}
pub fn compile_with_runtime(
    g: &Graph,
    root: Value,
    cache: &Path,
    runtime: &Runtime,
) -> Result<Executable> {
    gpu::compile_with_runtime(g, root, cache, &runtime.0)
}

/// Compile generated HIP source to an AMD code object without an attached GPU.
/// Architecture must be a ROCm target such as gfx1100 or gfx90a:xnack-.
pub fn compile_source(source: &str, architecture: &str) -> Result<Vec<u8>> {
    gpu::compile_hip_source(source, architecture)
}
