//! Backend selection for model stages and training; tensors remain host-owned at the API.
use super::{
    cpu, gpu,
    pop::{Error, Result},
    source,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu(gpu::Backend, usize),
}
impl Device {
    pub fn parse(selector: &str) -> Result<Self> {
        if matches!(selector, "cpu" | "cpu:0" | "c" | "c:0") {
            return Ok(Self::Cpu);
        }
        let (backend, index) = gpu::device(selector)?;
        Ok(Self::Gpu(backend, index))
    }
    pub fn validate_target(self, target: cpu::CpuTarget) -> Result<()> {
        if self != Self::Cpu && target != cpu::CpuTarget::Generic {
            return Err(Error("--cpu-target applies to the CPU backend".into()));
        }
        target.validate()
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "C",
            Self::Gpu(backend, _) => backend.label(),
        }
    }
    pub fn source_extension(self) -> &'static str {
        match self {
            Self::Cpu => "c",
            Self::Gpu(backend, _) => backend.source_extension(),
        }
    }
}

pub enum Executable {
    Cpu(cpu::Executable),
    Gpu(gpu::Executable),
}
impl Executable {
    pub fn run_with_threads(
        &self,
        inputs: &[cpu::Tensor],
        threads: usize,
    ) -> Result<Vec<cpu::Tensor>> {
        match self {
            Self::Cpu(e) => e.run_with_threads(inputs, threads),
            Self::Gpu(e) => e.run(inputs),
        }
    }
    /// GPU execution returns host wall time to the caller; kernel counters require event profiling.
    pub fn run_profiled(
        &self,
        inputs: &[cpu::Tensor],
        threads: usize,
    ) -> Result<(Vec<cpu::Tensor>, Option<Vec<u64>>)> {
        match self {
            Self::Cpu(e) => {
                let r = e.run_profiled(inputs, threads)?;
                Ok((r.outputs, Some(r.counters)))
            }
            Self::Gpu(e) => Ok((e.run(inputs)?, None)),
        }
    }
    pub fn source_path(&self) -> &Path {
        match self {
            Self::Cpu(e) => &e.source_path,
            Self::Gpu(e) => &e.source_path,
        }
    }
    pub fn metadata(&self) -> serde_json::Value {
        match self {
            Self::Cpu(e) => e.profile_metadata.as_ref().unwrap().clone(),
            Self::Gpu(e) => {
                serde_json::json!({"backend": e.backend().tag(), "kernel_count":e.kernel_count(), "workspace_bytes":e.workspace_bytes(), "device_profiling":false, "kernels":[]})
            }
        }
    }
    pub fn kernel_count(&self) -> usize {
        match self {
            Self::Cpu(e) => e.profile_metadata.as_ref().unwrap()["kernels"]
                .as_array()
                .unwrap()
                .len(),
            Self::Gpu(e) => e.kernel_count(),
        }
    }
    pub fn build_info(&self) -> serde_json::Value {
        match self {
            Self::Cpu(e) => {
                let mut info = serde_json::to_value(&e.build_info).unwrap();
                info["cache_hit"] = serde_json::json!(e.cache_hit);
                info
            }
            Self::Gpu(e) => {
                serde_json::json!({"backend":e.backend().tag(),"device":e.device_name,"cache_hit":e.cache_hit})
            }
        }
    }
}
pub fn compile_profiled(
    program: &source::Program,
    device: Device,
    options: &cpu::BuildOptions,
) -> Result<Executable> {
    device.validate_target(options.cpu_target)?;
    match device {
        Device::Cpu => Ok(Executable::Cpu(cpu::compile_profiled_with_options(
            program,
            Path::new(".cache/pup/cpu"),
            options,
        )?)),
        Device::Gpu(backend, index) => {
            let runtime = gpu::Runtime::new(backend, index)?;
            let cache = PathBuf::from(format!(".cache/pup/{}", backend.tag()));
            Ok(Executable::Gpu(gpu::compile_with_runtime(
                &program.graph,
                program.root,
                &cache,
                &runtime,
            )?))
        }
    }
}
