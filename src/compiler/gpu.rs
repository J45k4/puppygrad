//! Shared CUDA/HIP execution: residency, state, checked launches and graph replay.
use super::{
    cpu::Tensor,
    pop::{numel, DType, Error, Graph, Result, Value},
};
use std::{
    cell::{Cell, RefCell},
    collections::{hash_map::DefaultHasher, HashMap},
    ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
    hash::{Hash, Hasher},
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
};

#[path = "cuda_graph.rs"]
mod replay;

/// Generated GPU backend; model graphs and buffer ownership are shared.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Backend {
    Cuda,
    Hip,
}
impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Hip => "HIP",
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }
    pub fn env(self, suffix: &str) -> String {
        format!("PUPPYGRAD_{}_{suffix}", self.label())
    }
    pub fn source_extension(self) -> &'static str {
        match self {
            Self::Cuda => "cu",
            Self::Hip => "hip",
        }
    }
    fn binary_extension(self) -> &'static str {
        match self {
            Self::Cuda => "ptx",
            Self::Hip => "hsaco",
        }
    }
}
/// Parse a GPU selector without loading either vendor's libraries.
pub fn device(device: &str) -> Result<(Backend, usize)> {
    for backend in [Backend::Cuda, Backend::Hip] {
        if device == backend.tag() {
            return Ok((backend, 0));
        }
        if let Some(index) = device.strip_prefix(&format!("{}:", backend.tag())) {
            return index.parse().map(|i| (backend, i)).map_err(|_| {
                Error(format!(
                    "invalid {} device {device:?}; expected {}:<index>",
                    backend.label(),
                    backend.tag()
                ))
            });
        }
    }
    Err(Error(format!(
        "unsupported device {device:?}; use --device cpu, cuda:<index> or hip:<index>"
    )))
}
fn load_library(variable: &str, defaults: &[&str], name: &str) -> Result<libloading::Library> {
    let paths = std::env::var_os(variable).map_or_else(
        || defaults.iter().map(PathBuf::from).collect::<Vec<_>>(),
        |p| vec![p.into()],
    );
    let mut errors = Vec::new();
    for path in paths {
        match unsafe { libloading::Library::new(&path) } {
            Ok(library) => return Ok(library),
            Err(e) => errors.push(format!("{}: {e}", path.display())),
        }
    }
    Err(Error(format!(
        "cannot load {name}; install it or set {variable} to its library path: {}",
        errors.join("; ")
    )))
}
fn rtc_options(backend: Backend, architecture: &str) -> Vec<String> {
    let mut options = vec![
        format!("--gpu-architecture={architecture}"),
        "--std=c++11".into(),
    ];
    match backend {
        Backend::Cuda => options.extend(
            [
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ]
            .map(String::from),
        ),
        Backend::Hip => options.extend(
            [
                "-ffp-contract=off",
                "-fno-fast-math",
                "-fno-unsafe-math-optimizations",
                "-fno-finite-math-only",
                "-fdenormal-fp-math=ieee",
            ]
            .map(String::from),
        ),
    }
    options
}

type Handle = *mut c_void;
type DevicePtr = u64;
#[derive(Clone)]
pub(crate) enum Binding {
    Input(usize),
    Output(usize),
    Arena(usize),
}
pub(crate) struct Kernel {
    pub names: Vec<String>,
    pub bindings: Vec<Binding>,
    pub elements: usize,
    pub blocks: usize,
    pub code: String,
}
pub(crate) struct Lowered {
    pub source: String,
    pub kernels: Vec<Kernel>,
    pub inputs: Vec<(usize, DType, usize)>,
    pub state_slots: std::collections::HashSet<usize>,
    pub outputs: Vec<(DType, Vec<usize>)>,
    pub workspace_bytes: usize,
    pub gemm_count: usize,
    pub row_fusion_count: usize,
    pub parallel_reduction_count: usize,
}
pub(crate) fn dtype_bytes(dt: DType) -> usize {
    match dt {
        DType::WeakFloat | DType::WeakInt => 8,
        DType::F32 | DType::I32 => 4,
        _ => 1,
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ParameterMemory {
    pub bytes: usize,
    pub state: bool,
    pub dtype: DType,
}
/// Device allocations selected by lowering, without loading NVRTC or weights.
/// Driver/module/graph storage is additional and needs reserved headroom.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MemoryPlan {
    pub parameters: std::collections::BTreeMap<usize, ParameterMemory>,
    pub output_buffers: Vec<usize>,
    pub workspace_bytes: usize,
}
impl MemoryPlan {
    pub(crate) fn from_lowered(x: &Lowered) -> Result<Self> {
        let parameters = x
            .inputs
            .iter()
            .map(|&(slot, dt, n)| {
                let bytes = n
                    .checked_mul(dtype_bytes(dt))
                    .ok_or_else(|| Error("GPU memory plan overflow".into()))?
                    .max(1);
                let state = x.state_slots.contains(&slot);
                if state && bytes > 2 * 1024 * 1024 * 1024 {
                    return Err(Error("state allocation exceeds 2 GiB".into()));
                }
                Ok((
                    slot,
                    ParameterMemory {
                        bytes,
                        state,
                        dtype: dt,
                    },
                ))
            })
            .collect::<Result<_>>()?;
        let output_buffers = x
            .outputs
            .iter()
            .map(|(dt, shape)| {
                numel(shape)?
                    .checked_mul(dtype_bytes(*dt))
                    .ok_or_else(|| Error("GPU memory plan overflow".into()))
                    .map(|n| n.max(1))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            parameters,
            output_buffers,
            workspace_bytes: x.workspace_bytes.max(1),
        })
    }
    /// Combine shapes that share one retained runtime; weights and state slots
    /// are counted once, scratch and output buffers retain their largest size.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        for (&slot, p) in &other.parameters {
            if let Some(old) = self.parameters.get_mut(&slot) {
                if old.state != p.state
                    || (p.state && (old.bytes != p.bytes || old.dtype != p.dtype))
                {
                    return Err(Error(
                        "retained state layout changes between execution shapes".into(),
                    ));
                }
                old.bytes = old.bytes.max(p.bytes);
            } else {
                self.parameters.insert(slot, p.clone());
            }
        }
        self.workspace_bytes = self.workspace_bytes.max(other.workspace_bytes);
        self.output_buffers
            .resize(self.output_buffers.len().max(other.output_buffers.len()), 0);
        for (old, &n) in self.output_buffers.iter_mut().zip(&other.output_buffers) {
            *old = (*old).max(n);
        }
        Ok(())
    }
    pub fn input_bytes(&self) -> usize {
        self.parameters
            .values()
            .filter(|p| !p.state)
            .map(|p| p.bytes)
            .fold(0, usize::saturating_add)
    }
    pub fn state_bytes(&self) -> usize {
        self.parameters
            .values()
            .filter(|p| p.state)
            .map(|p| p.bytes)
            .fold(0, usize::saturating_add)
    }
    pub fn output_bytes(&self) -> usize {
        self.output_buffers
            .iter()
            .copied()
            .fold(0, usize::saturating_add)
    }
    pub fn total_bytes(&self) -> usize {
        self.input_bytes()
            .saturating_add(self.state_bytes())
            .saturating_add(self.workspace_bytes)
            .saturating_add(self.output_bytes())
            .saturating_add(4)
    }
}
pub fn memory_plan(g: &Graph, root: Value) -> Result<MemoryPlan> {
    memory_plan_for_backend(g, root, Backend::Cuda)
}
pub fn memory_plan_for_backend(g: &Graph, root: Value, backend: Backend) -> Result<MemoryPlan> {
    MemoryPlan::from_lowered(&super::cpu::cuda_lower::emit_backend(g, root, backend)?)
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DeviceMemory {
    pub free_bytes: usize,
    pub total_bytes: usize,
}
unsafe fn symbol<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T> {
    lib.get::<T>(name)
        .map(|s| *s)
        .map_err(|e| Error(e.to_string()))
}
#[path = "gpu_driver.rs"]
mod driver;
use driver::Driver;

struct Current<'a>(&'a Driver);
impl Drop for Current<'_> {
    fn drop(&mut self) {
        unsafe {
            let mut popped = std::ptr::null_mut();
            self.0.pop(&mut popped);
        }
    }
}
struct Context {
    driver: Driver,
    handle: Handle,
    device: c_int,
    memory_limit: Cell<Option<usize>>,
    allocated_bytes: Cell<usize>,
}
impl Context {
    fn new(backend: Backend, index: usize) -> Result<(Self, String, String)> {
        let driver = Driver::load(backend)?;
        driver.check(unsafe { driver.init(0) }, "initialize")?;
        let mut device = 0;
        driver.check(
            unsafe {
                driver.device(
                    &mut device,
                    c_int::try_from(index)
                        .map_err(|_| Error("GPU device index too large".into()))?,
                )
            },
            "select device",
        )?;
        let mut name = [0 as c_char; 256];
        driver.check(
            unsafe { driver.name(name.as_mut_ptr(), 256, device) },
            "device name",
        )?;
        let architecture = driver.architecture(device)?;
        let mut handle = std::ptr::null_mut();
        driver.check(
            unsafe { driver.retain(&mut handle, device) },
            "retain context",
        )?;
        Ok((
            Self {
                driver,
                handle,
                device,
                memory_limit: Cell::new(None),
                allocated_bytes: Cell::new(0),
            },
            unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            architecture,
        ))
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.driver.release(self.device);
        }
    }
}
struct Memory {
    context: Rc<Context>,
    ptr: DevicePtr,
    bytes: usize,
}
impl Memory {
    fn new(context: &Rc<Context>, bytes: usize) -> Result<Self> {
        let bytes = bytes.max(1);
        let total = context
            .allocated_bytes
            .get()
            .checked_add(bytes)
            .ok_or_else(|| Error("GPU allocation size overflow".into()))?;
        check_buffer_limit(total, context.memory_limit.get())?;
        let driver = &context.driver;
        let mut ptr = 0;
        driver.check(
            unsafe { driver.alloc(&mut ptr, bytes.max(1)) },
            "allocate buffer",
        )?;
        context.allocated_bytes.set(total);
        Ok(Self {
            context: context.clone(),
            ptr,
            bytes,
        })
    }
}
impl Drop for Memory {
    fn drop(&mut self) {
        // Buffers may outlive an executable and must free under their own context.
        if let Ok(_current) = self.context.driver.enter(self.context.handle) {
            unsafe {
                self.context.driver.free(self.ptr);
            }
        }
        self.context.allocated_bytes.set(
            self.context
                .allocated_bytes
                .get()
                .saturating_sub(self.bytes),
        );
    }
}

fn check_buffer_limit(bytes: usize, limit: Option<usize>) -> Result<()> {
    if let Some(limit) = limit.filter(|limit| bytes > *limit) {
        return Err(Error(format!(
            "GPU model buffers need {bytes} bytes, exceeding --max-memory {limit} bytes"
        )));
    }
    Ok(())
}

/// Shared device residency for related executables (for example static shapes).
/// Dropping the last runtime/executable owner releases all retained buffers.
#[derive(Clone)]
pub struct Runtime(Rc<RuntimeInner>);
struct RuntimeInner {
    context: Rc<Context>,
    device_name: String,
    architecture: String,
    buffers: RefCell<Buffers>,
    graph_replay: Cell<bool>,
}
#[derive(Default)]
struct Buffers {
    arena: Option<Memory>,
    error: Option<Memory>,
    outputs: Vec<Memory>,
    inputs: HashMap<usize, ResidentInput>,
    states: HashMap<usize, (DType, usize, Memory)>,
    allocations: u64,
    execution: ExecutionStats,
    input_uploads: u64,
    input_uploaded_bytes: u64,
}
struct ResidentInput {
    memory: Memory,
    // Retain the Arc, not just its address: this prevents address reuse and
    // forces safe Arc::make_mut callers to create a new allocation on updates.
    host: Option<Tensor>,
}
#[derive(Clone, Copy, Debug)]
pub struct ResidencyStats {
    pub allocations: u64,
    pub input_uploads: u64,
    pub input_uploaded_bytes: u64,
    pub resident_bytes: usize,
    pub input_bytes: usize,
    pub arena_bytes: usize,
    pub state_bytes: usize,
}
/// Host submission counters; kernel_count still counts GPU kernel work.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionStats {
    pub graph_builds: u64,
    pub graph_launches: u64,
    pub direct_kernel_launches: u64,
}
impl Runtime {
    /// Limit this runtime's buffers, including temporary overlap during growth.
    /// Driver modules and other processes are outside this model buffer budget.
    pub fn set_memory_limit(&self, limit: Option<usize>) -> Result<()> {
        if limit == Some(0) {
            return Err(Error("memory limit must be positive".into()));
        }
        check_buffer_limit(self.0.context.allocated_bytes.get(), limit)?;
        self.0.context.memory_limit.set(limit);
        Ok(())
    }
    pub fn backend(&self) -> Backend {
        self.0.context.driver.backend()
    }
    pub fn memory_info(&self) -> Result<DeviceMemory> {
        let context = &self.0.context;
        let _current = context.driver.enter(context.handle)?;
        let (mut free_bytes, mut total_bytes) = (0, 0);
        context.driver.check(
            unsafe {
                context
                    .driver
                    .memory_info(&mut free_bytes, &mut total_bytes)
            },
            "memory info",
        )?;
        Ok(DeviceMemory {
            free_bytes,
            total_bytes,
        })
    }
    /// Account for buffers already retained by other shapes before allocating.
    /// This is a preflight estimate; other GPU processes may allocate afterwards.
    pub fn check_memory(&self, plan: &MemoryPlan, reserve_bytes: usize) -> Result<()> {
        let info = self.memory_info()?;
        let b = self.0.buffers.borrow();
        let mut extra = plan
            .workspace_bytes
            .saturating_sub(b.arena.as_ref().map_or(0, |m| m.bytes));
        // A growing allocation is created before its old buffer is dropped.
        // Net growth alone would undercount this transient allocation peak.
        let mut overlap = b
            .arena
            .as_ref()
            .filter(|m| m.bytes < plan.workspace_bytes)
            .map_or(0, |m| m.bytes);
        extra =
            extra.saturating_add(4usize.saturating_sub(b.error.as_ref().map_or(0, |m| m.bytes)));
        for (&slot, p) in &plan.parameters {
            let owned = if p.state {
                b.states.get(&slot).map_or(0, |(_, _, m)| m.bytes)
            } else {
                b.inputs.get(&slot).map_or(0, |x| x.memory.bytes)
            };
            let replaces = if p.state {
                b.states
                    .get(&slot)
                    .is_some_and(|(dt, _, m)| *dt != p.dtype || m.bytes != p.bytes)
            } else {
                owned < p.bytes
            };
            if replaces {
                overlap = overlap.max(owned.min(p.bytes));
            }
            extra = extra.saturating_add(p.bytes.saturating_sub(owned));
        }
        for (slot, &bytes) in plan.output_buffers.iter().enumerate() {
            let owned = b.outputs.get(slot).map_or(0, |m| m.bytes);
            if owned < bytes {
                overlap = overlap.max(owned);
            }
            extra = extra
                .saturating_add(bytes.saturating_sub(b.outputs.get(slot).map_or(0, |m| m.bytes)));
        }
        extra = extra.saturating_add(overlap);
        check_buffer_limit(
            self.0.context.allocated_bytes.get().saturating_add(extra),
            self.0.context.memory_limit.get(),
        )?;
        if extra.saturating_add(reserve_bytes) > info.free_bytes {
            return Err(Error(format!("insufficient free GPU memory: need {extra} additional bytes plus {reserve_bytes} reserved, available {}", info.free_bytes)));
        }
        Ok(())
    }
    /// Select graph replay or direct kernel launches without changing model source.
    pub fn set_graph_replay(&self, enabled: bool) {
        self.0.graph_replay.set(enabled);
    }
    pub fn execution_stats(&self) -> ExecutionStats {
        self.0.buffers.borrow().execution
    }
    pub fn reset_state(&self) -> Result<()> {
        let context = &self.0.context;
        let d = &context.driver;
        let _current = d.enter(context.handle)?;
        let buffers = self
            .0
            .buffers
            .try_borrow_mut()
            .map_err(|_| Error("GPU state is executing".into()))?;
        for (_, _, memory) in buffers.states.values() {
            d.check(
                unsafe { d.memset(memory.ptr, 0, memory.bytes) },
                "reset state",
            )?;
        }
        d.check(unsafe { d.synchronize() }, "synchronize state reset")
    }
    pub(crate) fn grow_state(
        &self,
        slot: usize,
        dtype: DType,
        old: &[usize],
        new: &[usize],
    ) -> Result<()> {
        if old == new {
            return Ok(());
        }
        let context = &self.0.context;
        let d = &context.driver;
        let _current = d.enter(context.handle)?;
        let mut buffers = self
            .0
            .buffers
            .try_borrow_mut()
            .map_err(|_| Error("GPU state is executing".into()))?;
        let Some((dt, n, memory)) = buffers.states.get(&slot) else {
            return Ok(());
        };
        if *dt != dtype || *n != numel(old)? {
            return Err(Error("retained state metadata changed".into()));
        }
        let mut original = vec![0u8; n * dtype_bytes(dtype)];
        if !original.is_empty() {
            d.check(
                unsafe { d.download(original.as_mut_ptr().cast(), memory.ptr, original.len()) },
                "read state for growth",
            )?;
        }
        let grown = super::state_resize::grow(&original, old, new, dtype_bytes(dtype))?;
        let memory = Memory::new(context, grown.len())?;
        if !grown.is_empty() {
            d.check(
                unsafe { d.upload(memory.ptr, grown.as_ptr().cast(), grown.len()) },
                "preserve grown state",
            )?;
        }
        buffers.states.insert(slot, (dtype, numel(new)?, memory));
        buffers.allocations += 1;
        Ok(())
    }
    pub fn new(backend: Backend, device: usize) -> Result<Self> {
        let (context, device_name, architecture) = Context::new(backend, device)?;
        Ok(Self(Rc::new(RuntimeInner {
            context: Rc::new(context),
            device_name,
            architecture,
            buffers: RefCell::new(Buffers::default()),
            graph_replay: Cell::new(std::env::var(backend.env("GRAPH")).as_deref() != Ok("0")),
        })))
    }
    /// Input counters exclude the four-byte device error flag reset.
    pub fn residency_stats(&self) -> ResidencyStats {
        let b = self.0.buffers.borrow();
        let input_bytes = b.inputs.values().map(|x| x.memory.bytes).sum::<usize>();
        let arena_bytes = b.arena.as_ref().map_or(0, |x| x.bytes);
        let state_bytes = b.states.values().map(|(_, _, m)| m.bytes).sum::<usize>();
        ResidencyStats {
            allocations: b.allocations,
            input_uploads: b.input_uploads,
            input_uploaded_bytes: b.input_uploaded_bytes,
            resident_bytes: input_bytes
                + state_bytes
                + arena_bytes
                + b.outputs.iter().map(|x| x.bytes).sum::<usize>()
                + b.error.as_ref().map_or(0, |x| x.bytes),
            input_bytes,
            arena_bytes,
            state_bytes,
        }
    }
}
fn same_tensor(a: &Tensor, b: &Tensor) -> bool {
    // dtype and length also distinguish zero-length and differently typed Arcs.
    a.dtype() == b.dtype() && a.len() == b.len() && a.ptr() == b.ptr()
}
fn grow(context: &Rc<Context>, memory: &mut Option<Memory>, bytes: usize) -> Result<bool> {
    if memory.as_ref().is_none_or(|m| m.bytes < bytes.max(1)) {
        *memory = Some(Memory::new(context, bytes)?);
        return Ok(true);
    }
    Ok(false)
}
struct Module<'a> {
    driver: &'a Driver,
    handle: Handle,
}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe {
            self.driver.unload_module(self.handle);
        }
    }
}

pub struct Executable {
    runtime: Runtime,
    module: Handle,
    functions: Vec<Handle>,
    lowered: Lowered,
    replay: RefCell<Option<replay::Replay>>,
    pub source_path: PathBuf,
    pub gemm_count: usize,
    pub cache_hit: bool,
    pub device_name: String,
    // Driver contexts are thread-local; prevent implicit cross-thread sharing.
    _local: PhantomData<Rc<()>>,
}
impl Drop for Executable {
    fn drop(&mut self) {
        // Recorded functions must be released before their PTX module.
        self.replay.get_mut().take();
        let context = &self.runtime.0.context;
        if let Ok(_current) = context.driver.enter(context.handle) {
            unsafe {
                context.driver.unload_module(self.module);
            }
        }
    }
}
/// Compile another shape/program sharing one device's retained input/scratch storage.
pub fn compile_with_runtime(
    g: &Graph,
    root: Value,
    cache: &Path,
    runtime: &Runtime,
) -> Result<Executable> {
    let backend = runtime.backend();
    let lowered = super::cpu::cuda_lower::emit_backend(g, root, backend)?;
    compile_lowered_with_runtime(lowered, cache, runtime)
}

/// Consume the lowering already used to validate a request's memory budget.
pub(crate) fn compile_lowered_with_runtime(
    lowered: Lowered,
    cache: &Path,
    runtime: &Runtime,
) -> Result<Executable> {
    let backend = runtime.backend();
    let context = &runtime.0.context;
    let architecture = &runtime.0.architecture;
    let nvrtc = Rtc::load(backend)?;
    let version = nvrtc.version()?;
    let mut hash = DefaultHasher::new();
    (
        "puppygrad-gpu-v1",
        backend,
        &lowered.source,
        &architecture,
        version,
    )
        .hash(&mut hash);
    std::fs::create_dir_all(cache).map_err(|e| Error(e.to_string()))?;
    let stem = format!("{:016x}", hash.finish());
    let source_path = cache.join(format!("{stem}.{}", backend.source_extension()));
    let ptx_path = cache.join(format!("{stem}.{}", backend.binary_extension()));
    let compiler_version = format!("{}.{}", version.0, version.1);
    let compiler_options = serde_json::to_string(&rtc_options(backend, architecture))
        .map_err(|e| Error(e.to_string()))?;
    let identity = super::kernel_cache::Identity {
        key: &stem,
        backend: backend.tag(),
        architecture,
        compiler_version: &compiler_version,
        compiler_options: &compiler_options,
    };
    // Cache bookkeeping is disposable. A missing/unavailable index must not
    // prevent the existing file cache and GPU compiler from working.
    let index = match super::kernel_cache::Index::open(cache) {
        Ok(index) => Some(index),
        Err(error) => {
            crate::progress::warning(format!(
                "Kernel cache index unavailable: {error}; using file cache"
            ));
            None
        }
    };
    let cached = std::fs::read_to_string(&source_path)
        .ok()
        .filter(|s| s == &lowered.source)
        .and_then(|_| std::fs::read(&ptx_path).ok())
        .filter(|binary| {
            if binary.is_empty() {
                return false;
            }
            let Some(index) = &index else {
                return true;
            };
            let artifacts = super::kernel_cache::Artifacts::new(lowered.source.as_bytes(), binary);
            match index.allows(&identity, &artifacts) {
                Ok(allowed) => allowed,
                Err(error) => {
                    crate::progress::warning(format!(
                        "Kernel cache metadata lookup failed: {error}; recompiling module"
                    ));
                    false
                }
            }
        });
    let cache_hit = cached.is_some();
    let started = std::time::Instant::now();
    crate::progress::emit(format!(
        "{} {} module {stem} · {} kernels · {architecture}",
        if cache_hit {
            "Loading cached"
        } else {
            "Compiling"
        },
        backend.label(),
        lowered.kernels.len()
    ));
    let ptx = if let Some(ptx) = cached {
        ptx
    } else {
        let ptx = nvrtc.compile(&lowered.source, architecture)?;
        write_atomic(&source_path, lowered.source.as_bytes())?;
        write_atomic(&ptx_path, &ptx)?;
        ptx
    };
    let artifacts = super::kernel_cache::Artifacts::new(lowered.source.as_bytes(), &ptx);
    let current = context.driver.enter(context.handle)?;
    let driver = &context.driver;
    // CUDA loads zero-terminated PTX; HIP loads an ELF code object containing NUL bytes.
    let module_image = match backend {
        Backend::Cuda => CString::new(ptx.strip_suffix(&[0]).unwrap_or(&ptx))
            .map_err(|_| Error("invalid cached PTX".into()))?
            .into_bytes_with_nul(),
        Backend::Hip => ptx,
    };
    if module_image.is_empty() {
        return Err(Error("empty cached GPU module".into()));
    }
    let mut handle = std::ptr::null_mut();
    driver.check(
        unsafe { driver.load_module(&mut handle, module_image.as_ptr().cast()) },
        "load GPU module",
    )?;
    let module = Module { driver, handle };
    let mut functions = Vec::new();
    for id in 0..lowered.kernels.len() {
        let name = CString::new(format!("kernel{id}")).unwrap();
        let mut function = std::ptr::null_mut();
        driver.check(
            unsafe { driver.function(&mut function, handle, name.as_ptr()) },
            "find kernel",
        )?;
        functions.push(function);
    }
    // Register only modules whose image and all expected functions loaded.
    // Timestamps/counts describe module loads, not individual GPU launches.
    if let Some(index) = &index {
        if let Err(error) = index.record(
            &identity,
            &artifacts,
            source_path.file_name().unwrap().to_str().unwrap(),
            ptx_path.file_name().unwrap().to_str().unwrap(),
            lowered.kernels.len(),
            lowered.gemm_count,
            cache_hit,
        ) {
            crate::progress::warning(format!("Kernel cache metadata update failed: {error}"));
        }
    }
    std::mem::forget(module);
    drop(current);
    crate::progress::emit(format!(
        "{} module ready · {:.3}s{}",
        backend.label(),
        started.elapsed().as_secs_f64(),
        if cache_hit {
            " · cache hit"
        } else {
            " · compiled"
        }
    ));
    let gemm_count = lowered.gemm_count;
    Ok(Executable {
        runtime: runtime.clone(),
        module: handle,
        functions,
        lowered,
        replay: RefCell::new(None),
        source_path,
        gemm_count,
        cache_hit,
        device_name: runtime.0.device_name.clone(),
        _local: PhantomData,
    })
}
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = path.with_extension(format!("tmp-{}-{unique}", std::process::id()));
    std::fs::write(&tmp, bytes)
        .and_then(|_| std::fs::rename(&tmp, path))
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            Error(e.to_string())
        })
}
impl Executable {
    pub fn backend(&self) -> Backend {
        self.runtime.backend()
    }
    pub fn execution_stats(&self) -> ExecutionStats {
        self.runtime.execution_stats()
    }
    pub fn residency_stats(&self) -> ResidencyStats {
        self.runtime.residency_stats()
    }
    pub fn row_fusion_count(&self) -> usize {
        self.lowered.row_fusion_count
    }
    pub fn parallel_reduction_count(&self) -> usize {
        self.lowered.parallel_reduction_count
    }
    pub fn kernel_count(&self) -> usize {
        self.lowered
            .kernels
            .iter()
            .filter(|k| k.elements > 0)
            .count()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.lowered.workspace_bytes
    }
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        for &(slot, dt, size) in &self.lowered.inputs {
            if self.lowered.state_slots.contains(&slot) {
                continue;
            }
            let x = inputs
                .get(slot)
                .ok_or_else(|| Error(format!("missing input slot {slot}")))?;
            if x.dtype() != dt || x.len() != size {
                return Err(Error(format!(
                    "input slot {slot}: expected {dt:?}[{size}], got {:?}[{}]",
                    x.dtype(),
                    x.len()
                )));
            }
        }
        let context = &self.runtime.0.context;
        let d = &context.driver;
        let _current = d.enter(context.handle)?;
        let mut buffers = self
            .runtime
            .0
            .buffers
            .try_borrow_mut()
            .map_err(|_| Error("GPU runtime is already executing".into()))?;
        buffers.allocations += u64::from(grow(
            context,
            &mut buffers.arena,
            self.lowered.workspace_bytes,
        )?);
        buffers.allocations += u64::from(grow(context, &mut buffers.error, 4)?);
        let error = buffers.error.as_ref().unwrap().ptr;
        let arena = buffers.arena.as_ref().unwrap().ptr;
        d.check(
            unsafe { d.upload(error, (&0i32 as *const i32).cast(), 4) },
            "initialize error flag",
        )?;
        let mut device_inputs = HashMap::new();
        for &(slot, dt, size) in &self.lowered.inputs {
            let bytes = size
                .checked_mul(dtype_bytes(dt))
                .ok_or_else(|| Error("GPU input size overflow".into()))?;
            if self.lowered.state_slots.contains(&slot) {
                if bytes > 2 * 1024 * 1024 * 1024 {
                    return Err(Error("state allocation exceeds 2 GiB".into()));
                }
                if buffers
                    .states
                    .get(&slot)
                    .is_none_or(|(dtype, n, _)| *dtype != dt || *n != size)
                {
                    let memory = Memory::new(context, bytes)?;
                    d.check(
                        unsafe { d.memset(memory.ptr, 0, memory.bytes) },
                        "initialize state",
                    )?;
                    buffers.states.insert(slot, (dt, size, memory));
                    buffers.allocations += 1;
                }
                device_inputs.insert(slot, buffers.states[&slot].2.ptr);
                continue;
            }
            if buffers
                .inputs
                .get(&slot)
                .is_none_or(|x| x.memory.bytes < bytes.max(1))
            {
                let memory = Memory::new(context, bytes)?;
                buffers
                    .inputs
                    .insert(slot, ResidentInput { memory, host: None });
                buffers.allocations += 1;
            }
            let resident = buffers.inputs.get_mut(&slot).unwrap();
            if !resident
                .host
                .as_ref()
                .is_some_and(|x| same_tensor(x, &inputs[slot]))
            {
                // Invalidate before copying: a failed upload must never be reused.
                resident.host = None;
                if bytes > 0 {
                    d.check(
                        unsafe { d.upload(resident.memory.ptr, inputs[slot].ptr(), bytes) },
                        "upload input",
                    )?;
                }
                resident.host = Some(inputs[slot].clone());
                buffers.input_uploads += 1;
                buffers.input_uploaded_bytes += bytes as u64;
            }
            device_inputs.insert(slot, buffers.inputs[&slot].memory.ptr);
        }
        for (slot, (dt, sh)) in self.lowered.outputs.iter().enumerate() {
            let bytes = numel(sh)?
                .checked_mul(dtype_bytes(*dt))
                .ok_or_else(|| Error("GPU output size overflow".into()))?;
            if buffers.outputs.len() <= slot {
                buffers.outputs.push(Memory::new(context, bytes)?);
                buffers.allocations += 1;
            } else if buffers.outputs[slot].bytes < bytes.max(1) {
                buffers.outputs[slot] = Memory::new(context, bytes)?;
                buffers.allocations += 1;
            }
        }
        let device_outputs = &buffers.outputs;
        let mut submissions = ExecutionStats::default();
        let launched = (|| {
            if self.runtime.0.graph_replay.get() && self.kernel_count() > 0 {
                let mut cached = self.replay.borrow_mut();
                // Every allocation/replacement advances this epoch, even when
                // CUDA recycles a freed address. Other shapes share these buffers.
                if cached
                    .as_ref()
                    .is_none_or(|g| g.epoch != buffers.allocations)
                {
                    cached.take();
                    *cached = Some(replay::Replay::build(
                        context,
                        &self.lowered.kernels,
                        &self.functions,
                        &device_inputs,
                        device_outputs,
                        arena,
                        error,
                        buffers.allocations,
                    )?);
                    submissions.graph_builds += 1;
                }
                cached.as_ref().unwrap().launch()?;
                submissions.graph_launches += 1;
            } else {
                for (id, k) in self.lowered.kernels.iter().enumerate() {
                    if k.elements == 0 {
                        continue;
                    }
                    let mut args =
                        replay::Arguments::new(k, &device_inputs, device_outputs, arena, error)?;
                    let p = args.params(self.functions[id]);
                    d.check(
                        unsafe {
                            d.launch(
                                p.func,
                                p.grid_x,
                                p.grid_y,
                                p.grid_z,
                                p.block_x,
                                p.block_y,
                                p.block_z,
                                p.shared_bytes,
                                std::ptr::null_mut(),
                                p.arguments,
                                p.extra,
                            )
                        },
                        "launch kernel",
                    )?;
                    submissions.direct_kernel_launches += 1;
                }
            }
            Ok(())
        })();
        buffers.execution.graph_builds += submissions.graph_builds;
        buffers.execution.graph_launches += submissions.graph_launches;
        buffers.execution.direct_kernel_launches += submissions.direct_kernel_launches;
        let device_outputs = &buffers.outputs;
        // Synchronize even on launch failure before retained storage can be reused.
        let synchronized = d.check(unsafe { d.synchronize() }, "synchronize");
        launched?;
        synchronized?;
        let mut status = 0i32;
        d.check(
            unsafe { d.download((&mut status as *mut i32).cast(), error, 4) },
            "read error flag",
        )?;
        if status != 0 {
            return Err(Error("INDEX out of bounds".into()));
        }
        let mut outputs = Vec::new();
        for ((dt, sh), mem) in self.lowered.outputs.iter().zip(device_outputs) {
            let n = numel(sh)?;
            macro_rules! download {
                ($t:ty,$variant:ident) => {{
                    let mut v = vec![0 as $t; n];
                    if n > 0 {
                        d.check(
                            unsafe {
                                d.download(
                                    v.as_mut_ptr().cast(),
                                    mem.ptr,
                                    n * std::mem::size_of::<$t>(),
                                )
                            },
                            "download output",
                        )?;
                    }
                    Tensor::$variant(v.into())
                }};
            }
            outputs.push(match dt {
                DType::F32 => download!(f32, F32),
                DType::I32 => download!(i32, I32),
                DType::U8 => download!(u8, U8),
                DType::Bool => download!(u8, Bool),
                _ => unreachable!(),
            });
        }
        Ok(outputs)
    }
}

struct Rtc {
    backend: Backend,
    library: libloading::Library,
}
impl Rtc {
    fn load(backend: Backend) -> Result<Self> {
        let library = match backend {
            Backend::Cuda => load_library(
                "PUPPYGRAD_NVRTC",
                &[
                    "libnvrtc.so",
                    "libnvrtc.so.12",
                    "libnvrtc.so.13",
                    "/opt/cuda/lib64/libnvrtc.so",
                    "/usr/local/cuda/lib64/libnvrtc.so",
                    ".cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so",
                    ".cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
                    ".cache/cuda-toolchain/nvidia/cuda_nvrtc/lib/libnvrtc.so.13",
                ],
                "NVRTC",
            )?,
            Backend::Hip => load_library(
                "PUPPYGRAD_HIPRTC",
                &[
                    "libhiprtc.so",
                    "/opt/rocm/lib/libhiprtc.so",
                    "/opt/rocm/lib64/libhiprtc.so",
                ],
                "HIPRTC",
            )?,
        };
        Ok(Self { backend, library })
    }
    fn api(&self, suffix: &str) -> CString {
        CString::new(format!(
            "{}{suffix}",
            match self.backend {
                Backend::Cuda => "nvrtc",
                Backend::Hip => "hiprtc",
            }
        ))
        .unwrap()
    }
    fn check(&self, code: c_int, operation: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(Error(format!(
                "{}RTC {operation} failed ({code})",
                match self.backend {
                    Backend::Cuda => "NV",
                    Backend::Hip => "HIP",
                }
            )))
        }
    }
    fn version(&self) -> Result<(i32, i32)> {
        unsafe {
            let f = symbol::<unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int>(
                &self.library,
                self.api("Version").as_bytes_with_nul(),
            )?;
            let mut major = 0;
            let mut minor = 0;
            self.check(f(&mut major, &mut minor), "version")?;
            Ok((major, minor))
        }
    }
    fn compile(&self, source: &str, architecture: &str) -> Result<Vec<u8>> {
        unsafe {
            let create =
                symbol::<
                    unsafe extern "C" fn(
                        *mut Handle,
                        *const c_char,
                        *const c_char,
                        c_int,
                        *const *const c_char,
                        *const *const c_char,
                    ) -> c_int,
                >(&self.library, self.api("CreateProgram").as_bytes_with_nul())?;
            let compile =
                symbol::<unsafe extern "C" fn(Handle, c_int, *const *const c_char) -> c_int>(
                    &self.library,
                    self.api("CompileProgram").as_bytes_with_nul(),
                )?;
            let destroy = symbol::<unsafe extern "C" fn(*mut Handle) -> c_int>(
                &self.library,
                self.api("DestroyProgram").as_bytes_with_nul(),
            )?;
            let log_size = symbol::<unsafe extern "C" fn(Handle, *mut usize) -> c_int>(
                &self.library,
                self.api("GetProgramLogSize").as_bytes_with_nul(),
            )?;
            let log = symbol::<unsafe extern "C" fn(Handle, *mut c_char) -> c_int>(
                &self.library,
                self.api("GetProgramLog").as_bytes_with_nul(),
            )?;
            let ptx_size = symbol::<unsafe extern "C" fn(Handle, *mut usize) -> c_int>(
                &self.library,
                self.api(match self.backend {
                    Backend::Cuda => "GetPTXSize",
                    Backend::Hip => "GetCodeSize",
                })
                .as_bytes_with_nul(),
            )?;
            let get_ptx = symbol::<unsafe extern "C" fn(Handle, *mut c_char) -> c_int>(
                &self.library,
                self.api(match self.backend {
                    Backend::Cuda => "GetPTX",
                    Backend::Hip => "GetCode",
                })
                .as_bytes_with_nul(),
            )?;
            let src = CString::new(source).map_err(|_| Error("NUL in GPU source".into()))?;
            let name =
                CString::new(format!("puppygrad.{}", self.backend.source_extension())).unwrap();
            let mut handle = std::ptr::null_mut();
            self.check(
                create(
                    &mut handle,
                    src.as_ptr(),
                    name.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                ),
                "create program",
            )?;
            struct Program {
                handle: Handle,
                destroy: unsafe extern "C" fn(*mut Handle) -> c_int,
            }
            impl Drop for Program {
                fn drop(&mut self) {
                    unsafe {
                        (self.destroy)(&mut self.handle);
                    }
                }
            }
            let program = Program { handle, destroy };
            let options = rtc_options(self.backend, architecture);
            let options = options
                .iter()
                .map(|s| CString::new(s.as_str()).unwrap())
                .collect::<Vec<_>>();
            let ptrs = options.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
            let status = compile(program.handle, ptrs.len() as c_int, ptrs.as_ptr());
            if status != 0 {
                let mut size = 0;
                self.check(log_size(handle, &mut size), "log size")?;
                let mut buffer = vec![0u8; size];
                self.check(log(handle, buffer.as_mut_ptr().cast()), "log")?;
                return Err(Error(format!(
                    "{} RTC compilation failed ({status}): {}",
                    self.backend.label(),
                    String::from_utf8_lossy(&buffer)
                )));
            }
            let mut size = 0;
            self.check(ptx_size(handle, &mut size), "PTX size")?;
            let mut ptx = vec![0; size];
            self.check(get_ptx(handle, ptx.as_mut_ptr().cast()), "PTX")?;
            Ok(ptx)
        }
    }
}
/// Offline HIPRTC compilation uses the same options as device execution.
pub(crate) fn compile_hip_source(source: &str, architecture: &str) -> Result<Vec<u8>> {
    if !architecture.starts_with("gfx")
        || !architecture
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b':' | b'+' | b'-'))
    {
        return Err(Error(format!(
            "invalid AMD HIP architecture {architecture:?}"
        )));
    }
    Rtc::load(Backend::Hip)?.compile(source, architecture)
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    #[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
    fn dropping_last_runtime_owner_releases_device_buffers() {
        dropping_buffers(Backend::Cuda);
    }
    #[test]
    #[ignore = "requires AMD GPU and HIPRTC"]
    fn hip_dropping_last_runtime_owner_releases_device_buffers() {
        dropping_buffers(Backend::Hip);
    }
    fn dropping_buffers(backend: Backend) {
        let runtime = Runtime::new(backend, 0).unwrap();
        let context = runtime.0.context.clone();
        let d = &context.driver;
        let _current = d.enter(context.handle).unwrap();
        let free_bytes = || {
            let (mut free, mut total) = (0, 0);
            d.check(
                unsafe { d.memory_info(&mut free, &mut total) },
                "memory info",
            )
            .unwrap();
            free
        };
        let program =
            crate::compiler::source::parse("x = param(0, f32, 16777216)\noutput reduce(x, add, 1)")
                .unwrap();
        let exe = compile_with_runtime(
            &program.graph,
            program.root,
            Path::new(".cache/pup/cuda-tests/drop"),
            &runtime,
        )
        .unwrap();
        let before = free_bytes();
        let inputs = [Tensor::F32(vec![0.; 16777216].into())];
        exe.run(&inputs).unwrap();
        let allocated = free_bytes();
        assert!(before.saturating_sub(allocated) >= 60 * 1024 * 1024);
        let weak = Rc::downgrade(&runtime.0);
        drop(runtime);
        assert!(weak.upgrade().is_some(), "executable retains its runtime");
        drop(exe);
        assert!(weak.upgrade().is_none(), "runtime has no hidden owners");
        let released = free_bytes();
        assert!(
            released.saturating_sub(allocated) >= 60 * 1024 * 1024,
            "VRAM must be freed while the primary context remains alive"
        );
    }
}
