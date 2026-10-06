//! Optional CUDA backend. Tensor math is generated CUDA C++, compiled to PTX
//! by NVRTC and executed through the dynamically loaded NVIDIA driver.
use super::{
    cpu::Tensor,
    pop::{numel, DType, Error, Graph, Result, Value},
};
use std::{
    cell::RefCell,
    collections::{hash_map::DefaultHasher, HashMap},
    ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
    hash::{Hash, Hasher},
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
};

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
}
pub(crate) fn dtype_bytes(dt: DType) -> usize {
    match dt {
        DType::WeakFloat | DType::WeakInt => 8,
        DType::F32 | DType::I32 => 4,
        _ => 1,
    }
}

/// Emit CUDA source without requiring NVIDIA libraries or a GPU.
pub fn emit(g: &Graph, root: Value) -> Result<(String, usize)> {
    let x = super::cpu::cuda_lower::emit(g, root)?;
    Ok((x.source, x.gemm_count))
}
pub fn device_index(device: &str) -> Result<usize> {
    if device == "cuda" {
        return Ok(0);
    }
    device
        .strip_prefix("cuda:")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            Error(format!(
                "invalid CUDA device {device:?}; expected cuda:<index>"
            ))
        })
}

unsafe fn symbol<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T> {
    lib.get::<T>(name)
        .map(|s| *s)
        .map_err(|e| Error(e.to_string()))
}
macro_rules! driver_api {
    ($($field:ident : $ty:ty => $name:literal),* $(,)?) => {
        struct Driver { _library:libloading::Library, $($field:$ty,)* }
        impl Driver { fn load()->Result<Self> {unsafe {
            let lib=libloading::Library::new("libcuda.so.1").map_err(|e|Error(format!("cannot load NVIDIA CUDA driver: {e}")))?;
            $(let $field=symbol::<$ty>(&lib,concat!($name,"\0").as_bytes())?;)*
            Ok(Self {_library:lib,$($field,)*})
        }}}
    }
}
driver_api! {
    init:unsafe extern "C" fn(c_uint)->c_int => "cuInit",
    device:unsafe extern "C" fn(*mut c_int,c_int)->c_int => "cuDeviceGet",
    name:unsafe extern "C" fn(*mut c_char,c_int,c_int)->c_int => "cuDeviceGetName",
    attribute:unsafe extern "C" fn(*mut c_int,c_int,c_int)->c_int => "cuDeviceGetAttribute",
    retain:unsafe extern "C" fn(*mut Handle,c_int)->c_int => "cuDevicePrimaryCtxRetain",
    release:unsafe extern "C" fn(c_int)->c_int => "cuDevicePrimaryCtxRelease_v2",
    push:unsafe extern "C" fn(Handle)->c_int => "cuCtxPushCurrent_v2",
    pop:unsafe extern "C" fn(*mut Handle)->c_int => "cuCtxPopCurrent_v2",
    load_module:unsafe extern "C" fn(*mut Handle,*const c_void)->c_int => "cuModuleLoadData",
    unload_module:unsafe extern "C" fn(Handle)->c_int => "cuModuleUnload",
    function:unsafe extern "C" fn(*mut Handle,Handle,*const c_char)->c_int => "cuModuleGetFunction",
    alloc:unsafe extern "C" fn(*mut DevicePtr,usize)->c_int => "cuMemAlloc_v2",
    free:unsafe extern "C" fn(DevicePtr)->c_int => "cuMemFree_v2",
    upload:unsafe extern "C" fn(DevicePtr,*const c_void,usize)->c_int => "cuMemcpyHtoD_v2",
    memset:unsafe extern "C" fn(DevicePtr,u8,usize)->c_int => "cuMemsetD8_v2",
    download:unsafe extern "C" fn(*mut c_void,DevicePtr,usize)->c_int => "cuMemcpyDtoH_v2",
    launch:unsafe extern "C" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut *mut c_void,*mut *mut c_void)->c_int => "cuLaunchKernel",
    synchronize:unsafe extern "C" fn()->c_int => "cuCtxSynchronize",
    error_string:unsafe extern "C" fn(c_int,*mut *const c_char)->c_int => "cuGetErrorString",
}
impl Driver {
    fn check(&self, code: c_int, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut message = std::ptr::null();
        unsafe {
            (self.error_string)(code, &mut message);
        }
        let detail = if message.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(message) }.to_string_lossy()
        };
        Err(Error(format!("CUDA {operation}: {detail} (error {code})")))
    }
    fn enter(&self, context: Handle) -> Result<Current<'_>> {
        self.check(unsafe { (self.push)(context) }, "push context")?;
        Ok(Current(self))
    }
}
struct Current<'a>(&'a Driver);
impl Drop for Current<'_> {
    fn drop(&mut self) {
        unsafe {
            let mut popped = std::ptr::null_mut();
            (self.0.pop)(&mut popped);
        }
    }
}
struct Context {
    driver: Driver,
    handle: Handle,
    device: c_int,
}
impl Context {
    fn new(index: usize) -> Result<(Self, String, String)> {
        let driver = Driver::load()?;
        driver.check(unsafe { (driver.init)(0) }, "initialize")?;
        let mut device = 0;
        driver.check(
            unsafe {
                (driver.device)(
                    &mut device,
                    c_int::try_from(index)
                        .map_err(|_| Error("CUDA device index too large".into()))?,
                )
            },
            "select device",
        )?;
        let mut name = [0 as c_char; 256];
        driver.check(
            unsafe { (driver.name)(name.as_mut_ptr(), 256, device) },
            "device name",
        )?;
        let mut major = 0;
        let mut minor = 0;
        driver.check(
            unsafe { (driver.attribute)(&mut major, 75, device) },
            "compute capability major",
        )?;
        driver.check(
            unsafe { (driver.attribute)(&mut minor, 76, device) },
            "compute capability minor",
        )?;
        let mut handle = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.retain)(&mut handle, device) },
            "retain context",
        )?;
        Ok((
            Self {
                driver,
                handle,
                device,
            },
            unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            format!("compute_{major}{minor}"),
        ))
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.driver.release)(self.device);
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
        let driver = &context.driver;
        let mut ptr = 0;
        driver.check(
            unsafe { (driver.alloc)(&mut ptr, bytes.max(1)) },
            "allocate buffer",
        )?;
        Ok(Self {
            context: context.clone(),
            ptr,
            bytes: bytes.max(1),
        })
    }
}
impl Drop for Memory {
    fn drop(&mut self) {
        // Buffers may outlive an executable and must free under their own context.
        if let Ok(_current) = self.context.driver.enter(self.context.handle) {
            unsafe {
                (self.context.driver.free)(self.ptr);
            }
        }
    }
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
}
#[derive(Default)]
struct Buffers {
    arena: Option<Memory>,
    error: Option<Memory>,
    outputs: Vec<Memory>,
    inputs: HashMap<usize, ResidentInput>,
    states: HashMap<usize, (DType, usize, Memory)>,
    allocations: u64,
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
impl Runtime {
    pub fn reset_state(&self) -> Result<()> {
        let context = &self.0.context;
        let d = &context.driver;
        let _current = d.enter(context.handle)?;
        let buffers = self
            .0
            .buffers
            .try_borrow_mut()
            .map_err(|_| Error("CUDA state is executing".into()))?;
        for (_, _, memory) in buffers.states.values() {
            d.check(
                unsafe { (d.memset)(memory.ptr, 0, memory.bytes) },
                "reset state",
            )?;
        }
        d.check(unsafe { (d.synchronize)() }, "synchronize state reset")
    }
    pub fn new(device: usize) -> Result<Self> {
        let (context, device_name, architecture) = Context::new(device)?;
        Ok(Self(Rc::new(RuntimeInner {
            context: Rc::new(context),
            device_name,
            architecture,
            buffers: RefCell::new(Buffers::default()),
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
            (self.driver.unload_module)(self.handle);
        }
    }
}

pub struct Executable {
    runtime: Runtime,
    module: Handle,
    functions: Vec<Handle>,
    lowered: Lowered,
    pub source_path: PathBuf,
    pub gemm_count: usize,
    pub cache_hit: bool,
    pub device_name: String,
    // Driver contexts are thread-local; prevent implicit cross-thread sharing.
    _local: PhantomData<Rc<()>>,
}
impl Drop for Executable {
    fn drop(&mut self) {
        let context = &self.runtime.0.context;
        if let Ok(_current) = context.driver.enter(context.handle) {
            unsafe {
                (context.driver.unload_module)(self.module);
            }
        }
    }
}
pub fn compile(g: &Graph, root: Value, cache: &Path, device: usize) -> Result<Executable> {
    compile_with_runtime(g, root, cache, &Runtime::new(device)?)
}
/// Compile another shape/program sharing one device's retained input/scratch storage.
pub fn compile_with_runtime(
    g: &Graph,
    root: Value,
    cache: &Path,
    runtime: &Runtime,
) -> Result<Executable> {
    let lowered = super::cpu::cuda_lower::emit(g, root)?;
    let context = &runtime.0.context;
    let architecture = &runtime.0.architecture;
    let nvrtc = Nvrtc::load()?;
    let version = nvrtc.version()?;
    let mut hash = DefaultHasher::new();
    ("puppygrad-cuda-v1", &lowered.source, &architecture, version).hash(&mut hash);
    std::fs::create_dir_all(cache).map_err(|e| Error(e.to_string()))?;
    let stem = format!("{:016x}", hash.finish());
    let source_path = cache.join(format!("{stem}.cu"));
    let ptx_path = cache.join(format!("{stem}.ptx"));
    let cached = std::fs::read_to_string(&source_path)
        .ok()
        .filter(|s| s == &lowered.source)
        .and_then(|_| std::fs::read(&ptx_path).ok());
    let cache_hit = cached.is_some();
    let ptx = if let Some(ptx) = cached {
        ptx
    } else {
        let ptx = nvrtc.compile(&lowered.source, &architecture)?;
        write_atomic(&source_path, lowered.source.as_bytes())?;
        write_atomic(&ptx_path, &ptx)?;
        ptx
    };
    let current = context.driver.enter(context.handle)?;
    let driver = &context.driver;
    let ptx = CString::new(ptx.strip_suffix(&[0]).unwrap_or(&ptx))
        .map_err(|_| Error("invalid cached PTX".into()))?;
    let mut handle = std::ptr::null_mut();
    driver.check(
        unsafe { (driver.load_module)(&mut handle, ptx.as_ptr().cast()) },
        "load PTX module",
    )?;
    let module = Module { driver, handle };
    let mut functions = Vec::new();
    for id in 0..lowered.kernels.len() {
        let name = CString::new(format!("kernel{id}")).unwrap();
        let mut function = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.function)(&mut function, handle, name.as_ptr()) },
            "find kernel",
        )?;
        functions.push(function);
    }
    std::mem::forget(module);
    drop(current);
    let gemm_count = lowered.gemm_count;
    Ok(Executable {
        runtime: runtime.clone(),
        module: handle,
        functions,
        lowered,
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
    pub fn residency_stats(&self) -> ResidencyStats {
        self.runtime.residency_stats()
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
            .map_err(|_| Error("CUDA runtime is already executing".into()))?;
        buffers.allocations += u64::from(grow(
            context,
            &mut buffers.arena,
            self.lowered.workspace_bytes,
        )?);
        buffers.allocations += u64::from(grow(context, &mut buffers.error, 4)?);
        let error = buffers.error.as_ref().unwrap().ptr;
        let arena = buffers.arena.as_ref().unwrap().ptr;
        d.check(
            unsafe { (d.upload)(error, (&0i32 as *const i32).cast(), 4) },
            "initialize error flag",
        )?;
        let mut device_inputs = HashMap::new();
        for &(slot, dt, size) in &self.lowered.inputs {
            let bytes = size
                .checked_mul(dtype_bytes(dt))
                .ok_or_else(|| Error("CUDA input size overflow".into()))?;
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
                        unsafe { (d.memset)(memory.ptr, 0, memory.bytes) },
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
                        unsafe { (d.upload)(resident.memory.ptr, inputs[slot].ptr(), bytes) },
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
                .ok_or_else(|| Error("CUDA output size overflow".into()))?;
            if buffers.outputs.len() <= slot {
                buffers.outputs.push(Memory::new(context, bytes)?);
                buffers.allocations += 1;
            } else if buffers.outputs[slot].bytes < bytes.max(1) {
                buffers.outputs[slot] = Memory::new(context, bytes)?;
                buffers.allocations += 1;
            }
        }
        let device_outputs = &buffers.outputs;
        let launched = (|| {
            for (id, k) in self.lowered.kernels.iter().enumerate() {
                if k.elements == 0 {
                    continue;
                }
                let mut values = k
                    .bindings
                    .iter()
                    .map(|b| match b {
                        Binding::Input(slot) => device_inputs[slot],
                        Binding::Output(slot) => device_outputs[*slot].ptr,
                        Binding::Arena(offset) => arena + *offset as u64,
                    })
                    .chain(std::iter::once(error))
                    .collect::<Vec<_>>();
                let mut args = values
                    .iter_mut()
                    .map(|x| (x as *mut u64).cast::<c_void>())
                    .collect::<Vec<_>>();
                let grid = c_uint::try_from(k.blocks)
                    .map_err(|_| Error("CUDA launch exceeds grid limit".into()))?;
                d.check(
                    unsafe {
                        (d.launch)(
                            self.functions[id],
                            grid,
                            1,
                            1,
                            256,
                            1,
                            1,
                            0,
                            std::ptr::null_mut(),
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "launch kernel",
                )?;
            }
            Ok(())
        })();
        // Synchronize even on launch failure before retained storage can be reused.
        let synchronized = d.check(unsafe { (d.synchronize)() }, "synchronize");
        launched?;
        synchronized?;
        let mut status = 0i32;
        d.check(
            unsafe { (d.download)((&mut status as *mut i32).cast(), error, 4) },
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
                                (d.download)(
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

struct Nvrtc {
    library: libloading::Library,
}
impl Nvrtc {
    fn load() -> Result<Self> {
        let paths = if let Some(path) = std::env::var_os("PUPPYGRAD_NVRTC") {
            vec![PathBuf::from(path)]
        } else {
            vec![
                "libnvrtc.so".into(),
                "libnvrtc.so.12".into(),
                "libnvrtc.so.13".into(),
                "/opt/cuda/lib64/libnvrtc.so".into(),
                "/usr/local/cuda/lib64/libnvrtc.so".into(),
            ]
        };
        let mut errors = Vec::new();
        for path in paths {
            match unsafe { libloading::Library::new(&path) } {
                Ok(library) => return Ok(Self { library }),
                Err(e) => errors.push(e.to_string()),
            }
        }
        Err(Error(format!("cannot load NVRTC; install NVIDIA NVRTC or set PUPPYGRAD_NVRTC to its library path: {}",errors.join("; "))))
    }
    fn version(&self) -> Result<(i32, i32)> {
        unsafe {
            let f = symbol::<unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int>(
                &self.library,
                b"nvrtcVersion\0",
            )?;
            let mut major = 0;
            let mut minor = 0;
            nvcheck(f(&mut major, &mut minor), "version")?;
            Ok((major, minor))
        }
    }
    fn compile(&self, source: &str, architecture: &str) -> Result<Vec<u8>> {
        unsafe {
            let create = symbol::<
                unsafe extern "C" fn(
                    *mut Handle,
                    *const c_char,
                    *const c_char,
                    c_int,
                    *const *const c_char,
                    *const *const c_char,
                ) -> c_int,
            >(&self.library, b"nvrtcCreateProgram\0")?;
            let compile = symbol::<
                unsafe extern "C" fn(Handle, c_int, *const *const c_char) -> c_int,
            >(&self.library, b"nvrtcCompileProgram\0")?;
            let destroy = symbol::<unsafe extern "C" fn(*mut Handle) -> c_int>(
                &self.library,
                b"nvrtcDestroyProgram\0",
            )?;
            let log_size = symbol::<unsafe extern "C" fn(Handle, *mut usize) -> c_int>(
                &self.library,
                b"nvrtcGetProgramLogSize\0",
            )?;
            let log = symbol::<unsafe extern "C" fn(Handle, *mut c_char) -> c_int>(
                &self.library,
                b"nvrtcGetProgramLog\0",
            )?;
            let ptx_size = symbol::<unsafe extern "C" fn(Handle, *mut usize) -> c_int>(
                &self.library,
                b"nvrtcGetPTXSize\0",
            )?;
            let get_ptx = symbol::<unsafe extern "C" fn(Handle, *mut c_char) -> c_int>(
                &self.library,
                b"nvrtcGetPTX\0",
            )?;
            let src = CString::new(source).map_err(|_| Error("NUL in CUDA source".into()))?;
            let name = CString::new("puppygrad.cu").unwrap();
            let mut handle = std::ptr::null_mut();
            nvcheck(
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
            let options = [
                format!("--gpu-architecture={architecture}"),
                "--std=c++11".into(),
                "--fmad=false".into(),
                "--ftz=false".into(),
                "--prec-div=true".into(),
                "--prec-sqrt=true".into(),
            ];
            let options = options
                .iter()
                .map(|s| CString::new(s.as_str()).unwrap())
                .collect::<Vec<_>>();
            let ptrs = options.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
            let status = compile(program.handle, ptrs.len() as c_int, ptrs.as_ptr());
            if status != 0 {
                let mut size = 0;
                nvcheck(log_size(handle, &mut size), "log size")?;
                let mut buffer = vec![0u8; size];
                nvcheck(log(handle, buffer.as_mut_ptr().cast()), "log")?;
                return Err(Error(format!(
                    "NVRTC compilation failed ({status}): {}",
                    String::from_utf8_lossy(&buffer)
                )));
            }
            let mut size = 0;
            nvcheck(ptx_size(handle, &mut size), "PTX size")?;
            let mut ptx = vec![0; size];
            nvcheck(get_ptx(handle, ptx.as_mut_ptr().cast()), "PTX")?;
            Ok(ptx)
        }
    }
}
fn nvcheck(code: c_int, operation: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error(format!("NVRTC {operation} failed ({code})")))
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    #[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
    fn dropping_last_runtime_owner_releases_device_buffers() {
        let runtime = Runtime::new(0).unwrap();
        let context = runtime.0.context.clone();
        let d = &context.driver;
        let _current = d.enter(context.handle).unwrap();
        let mem_info = unsafe {
            symbol::<unsafe extern "C" fn(*mut usize, *mut usize) -> c_int>(
                &d._library,
                b"cuMemGetInfo_v2\0",
            )
            .unwrap()
        };
        let free_bytes = || {
            let (mut free, mut total) = (0, 0);
            d.check(unsafe { mem_info(&mut free, &mut total) }, "memory info")
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
