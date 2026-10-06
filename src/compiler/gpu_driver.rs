//! Dynamically loaded GPU APIs. HIP pointer and graph ABIs are converted explicitly.
// HIP ABI declarations follow ROCm/HIP include/hip/hip_runtime_api.h (R0600).
// Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
use super::*;
macro_rules! driver_api {
    ($($field:ident : $ty:ty => $name:literal),* $(,)?) => {
        pub(super) struct Cuda { pub(super) _library:libloading::Library, $(pub(super) $field:$ty,)* }
        impl Cuda { pub(super) fn load()->Result<Self> {unsafe {
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
    memory_info:unsafe extern "C" fn(*mut usize,*mut usize)->c_int => "cuMemGetInfo_v2",
    upload:unsafe extern "C" fn(DevicePtr,*const c_void,usize)->c_int => "cuMemcpyHtoD_v2",
    memset:unsafe extern "C" fn(DevicePtr,u8,usize)->c_int => "cuMemsetD8_v2",
    download:unsafe extern "C" fn(*mut c_void,DevicePtr,usize)->c_int => "cuMemcpyDtoH_v2",
    launch:unsafe extern "C" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut *mut c_void,*mut *mut c_void)->c_int => "cuLaunchKernel",
    graph_create:unsafe extern "C" fn(*mut Handle,c_uint)->c_int => "cuGraphCreate",
    graph_destroy:unsafe extern "C" fn(Handle)->c_int => "cuGraphDestroy",
    graph_add_kernel:unsafe extern "C" fn(*mut Handle,Handle,*const Handle,usize,*const replay::KernelParams)->c_int => "cuGraphAddKernelNode",
    graph_instantiate:unsafe extern "C" fn(*mut Handle,Handle,*mut Handle,*mut c_char,usize)->c_int => "cuGraphInstantiate_v2",
    graph_exec_destroy:unsafe extern "C" fn(Handle)->c_int => "cuGraphExecDestroy",
    graph_launch:unsafe extern "C" fn(Handle,Handle)->c_int => "cuGraphLaunch",
    synchronize:unsafe extern "C" fn()->c_int => "cuCtxSynchronize",
    error_string:unsafe extern "C" fn(c_int,*mut *const c_char)->c_int => "cuGetErrorString",
}

pub(super) struct Hip {
    _library: libloading::Library,
    init: unsafe extern "C" fn(c_uint) -> c_int,
    device: unsafe extern "C" fn(*mut c_int, c_int) -> c_int,
    name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    retain: unsafe extern "C" fn(*mut Handle, c_int) -> c_int,
    release: unsafe extern "C" fn(c_int) -> c_int,
    push: unsafe extern "C" fn(Handle) -> c_int,
    pop: unsafe extern "C" fn(*mut Handle) -> c_int,
    load_module: unsafe extern "C" fn(*mut Handle, *const c_void) -> c_int,
    unload_module: unsafe extern "C" fn(Handle) -> c_int,
    function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> c_int,
    alloc: unsafe extern "C" fn(*mut Handle, usize) -> c_int,
    free: unsafe extern "C" fn(Handle) -> c_int,
    memory_info: unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
    upload: unsafe extern "C" fn(Handle, *const c_void, usize) -> c_int,
    memset: unsafe extern "C" fn(Handle, c_int, usize) -> c_int,
    download: unsafe extern "C" fn(*mut c_void, Handle, usize) -> c_int,
    launch: unsafe extern "C" fn(
        Handle,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
    graph_create: unsafe extern "C" fn(*mut Handle, c_uint) -> c_int,
    graph_destroy: unsafe extern "C" fn(Handle) -> c_int,
    graph_add_kernel: unsafe extern "C" fn(
        *mut Handle,
        Handle,
        *const Handle,
        usize,
        *const HipKernelParams,
    ) -> c_int,
    graph_instantiate:
        unsafe extern "C" fn(*mut Handle, Handle, *mut Handle, *mut c_char, usize) -> c_int,
    graph_exec_destroy: unsafe extern "C" fn(Handle) -> c_int,
    graph_launch: unsafe extern "C" fn(Handle, Handle) -> c_int,
    synchronize: unsafe extern "C" fn() -> c_int,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    properties: unsafe extern "C" fn(*mut HipDeviceProperties, c_int) -> c_int,
}
impl Hip {
    fn load() -> Result<Self> {
        let lib = load_library(
            "PUPPYGRAD_HIP_RUNTIME",
            &[
                "libamdhip64.so",
                "/opt/rocm/lib/libamdhip64.so",
                "/opt/rocm/lib64/libamdhip64.so",
            ],
            "AMD HIP runtime",
        )?;
        unsafe {
            let init = symbol::<unsafe extern "C" fn(c_uint) -> c_int>(&lib, b"hipInit\0")?;
            let device = symbol::<unsafe extern "C" fn(*mut c_int, c_int) -> c_int>(
                &lib,
                b"hipDeviceGet\0",
            )?;
            let name = symbol::<unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int>(
                &lib,
                b"hipDeviceGetName\0",
            )?;
            let retain = symbol::<unsafe extern "C" fn(*mut Handle, c_int) -> c_int>(
                &lib,
                b"hipDevicePrimaryCtxRetain\0",
            )?;
            let release = symbol::<unsafe extern "C" fn(c_int) -> c_int>(
                &lib,
                b"hipDevicePrimaryCtxRelease\0",
            )?;
            let push =
                symbol::<unsafe extern "C" fn(Handle) -> c_int>(&lib, b"hipCtxPushCurrent\0")?;
            let pop =
                symbol::<unsafe extern "C" fn(*mut Handle) -> c_int>(&lib, b"hipCtxPopCurrent\0")?;
            let load_module = symbol::<unsafe extern "C" fn(*mut Handle, *const c_void) -> c_int>(
                &lib,
                b"hipModuleLoadData\0",
            )?;
            let unload_module =
                symbol::<unsafe extern "C" fn(Handle) -> c_int>(&lib, b"hipModuleUnload\0")?;
            let function = symbol::<
                unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> c_int,
            >(&lib, b"hipModuleGetFunction\0")?;
            let alloc =
                symbol::<unsafe extern "C" fn(*mut Handle, usize) -> c_int>(&lib, b"hipMalloc\0")?;
            let free = symbol::<unsafe extern "C" fn(Handle) -> c_int>(&lib, b"hipFree\0")?;
            let memory_info = symbol::<unsafe extern "C" fn(*mut usize, *mut usize) -> c_int>(
                &lib,
                b"hipMemGetInfo\0",
            )?;
            let upload = symbol::<unsafe extern "C" fn(Handle, *const c_void, usize) -> c_int>(
                &lib,
                b"hipMemcpyHtoD\0",
            )?;
            let memset = symbol::<unsafe extern "C" fn(Handle, c_int, usize) -> c_int>(
                &lib,
                b"hipMemset\0",
            )?;
            let download = symbol::<unsafe extern "C" fn(*mut c_void, Handle, usize) -> c_int>(
                &lib,
                b"hipMemcpyDtoH\0",
            )?;
            let launch = symbol::<
                unsafe extern "C" fn(
                    Handle,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    Handle,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> c_int,
            >(&lib, b"hipModuleLaunchKernel\0")?;
            let graph_create = symbol::<unsafe extern "C" fn(*mut Handle, c_uint) -> c_int>(
                &lib,
                b"hipGraphCreate\0",
            )?;
            let graph_destroy =
                symbol::<unsafe extern "C" fn(Handle) -> c_int>(&lib, b"hipGraphDestroy\0")?;
            let graph_add_kernel = symbol::<
                unsafe extern "C" fn(
                    *mut Handle,
                    Handle,
                    *const Handle,
                    usize,
                    *const HipKernelParams,
                ) -> c_int,
            >(&lib, b"hipGraphAddKernelNode\0")?;
            let graph_instantiate = symbol::<
                unsafe extern "C" fn(*mut Handle, Handle, *mut Handle, *mut c_char, usize) -> c_int,
            >(&lib, b"hipGraphInstantiate\0")?;
            let graph_exec_destroy =
                symbol::<unsafe extern "C" fn(Handle) -> c_int>(&lib, b"hipGraphExecDestroy\0")?;
            let graph_launch =
                symbol::<unsafe extern "C" fn(Handle, Handle) -> c_int>(&lib, b"hipGraphLaunch\0")?;
            let synchronize =
                symbol::<unsafe extern "C" fn() -> c_int>(&lib, b"hipDeviceSynchronize\0")?;
            let error_string = symbol::<unsafe extern "C" fn(c_int) -> *const c_char>(
                &lib,
                b"hipGetErrorString\0",
            )?;
            let properties = symbol(&lib, b"hipGetDevicePropertiesR0600\0")?;
            Ok(Self {
                _library: lib,
                properties,
                init,
                device,
                name,
                retain,
                release,
                push,
                pop,
                load_module,
                unload_module,
                function,
                alloc,
                free,
                memory_info,
                upload,
                memset,
                download,
                launch,
                graph_create,
                graph_destroy,
                graph_add_kernel,
                graph_instantiate,
                graph_exec_destroy,
                graph_launch,
                synchronize,
                error_string,
            })
        }
    }
}
#[allow(clippy::large_enum_variant)]
pub(super) enum Driver {
    Cuda(Cuda),
    Hip(Hip),
}
#[allow(clippy::too_many_arguments)] // FFI launch signatures are fixed by vendor APIs.
impl Driver {
    pub fn load(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cuda => Ok(Self::Cuda(Cuda::load()?)),
            Backend::Hip => {
                if !cfg!(all(target_os = "linux", target_pointer_width = "64")) {
                    return Err(Error("HIP requires 64-bit Linux".into()));
                }
                Ok(Self::Hip(Hip::load()?))
            }
        }
    }
    pub fn backend(&self) -> Backend {
        match self {
            Self::Cuda(_) => Backend::Cuda,
            Self::Hip(_) => Backend::Hip,
        }
    }
    pub unsafe fn init(&self, a0: c_uint) -> c_int {
        match self {
            Self::Cuda(d) => (d.init)(a0),
            Self::Hip(d) => (d.init)(a0),
        }
    }
    pub unsafe fn device(&self, a0: *mut c_int, a1: c_int) -> c_int {
        match self {
            Self::Cuda(d) => (d.device)(a0, a1),
            Self::Hip(d) => (d.device)(a0, a1),
        }
    }
    pub unsafe fn name(&self, a0: *mut c_char, a1: c_int, a2: c_int) -> c_int {
        match self {
            Self::Cuda(d) => (d.name)(a0, a1, a2),
            Self::Hip(d) => (d.name)(a0, a1, a2),
        }
    }
    pub unsafe fn retain(&self, a0: *mut Handle, a1: c_int) -> c_int {
        match self {
            Self::Cuda(d) => (d.retain)(a0, a1),
            Self::Hip(d) => (d.retain)(a0, a1),
        }
    }
    pub unsafe fn release(&self, a0: c_int) -> c_int {
        match self {
            Self::Cuda(d) => (d.release)(a0),
            Self::Hip(d) => (d.release)(a0),
        }
    }
    pub unsafe fn push(&self, a0: Handle) -> c_int {
        match self {
            Self::Cuda(d) => (d.push)(a0),
            Self::Hip(d) => (d.push)(a0),
        }
    }
    pub unsafe fn pop(&self, a0: *mut Handle) -> c_int {
        match self {
            Self::Cuda(d) => (d.pop)(a0),
            Self::Hip(d) => (d.pop)(a0),
        }
    }
    pub unsafe fn load_module(&self, a0: *mut Handle, a1: *const c_void) -> c_int {
        match self {
            Self::Cuda(d) => (d.load_module)(a0, a1),
            Self::Hip(d) => (d.load_module)(a0, a1),
        }
    }
    pub unsafe fn unload_module(&self, a0: Handle) -> c_int {
        match self {
            Self::Cuda(d) => (d.unload_module)(a0),
            Self::Hip(d) => (d.unload_module)(a0),
        }
    }
    pub unsafe fn function(&self, a0: *mut Handle, a1: Handle, a2: *const c_char) -> c_int {
        match self {
            Self::Cuda(d) => (d.function)(a0, a1, a2),
            Self::Hip(d) => (d.function)(a0, a1, a2),
        }
    }
    pub unsafe fn alloc(&self, a0: *mut DevicePtr, a1: usize) -> c_int {
        match self {
            Self::Cuda(d) => (d.alloc)(a0, a1),
            Self::Hip(d) => {
                let mut ptr = std::ptr::null_mut();
                let status = (d.alloc)(&mut ptr, a1);
                *a0 = ptr as DevicePtr;
                status
            }
        }
    }
    pub unsafe fn free(&self, a0: DevicePtr) -> c_int {
        match self {
            Self::Cuda(d) => (d.free)(a0),
            Self::Hip(d) => (d.free)(a0 as Handle),
        }
    }
    pub unsafe fn memory_info(&self, a0: *mut usize, a1: *mut usize) -> c_int {
        match self {
            Self::Cuda(d) => (d.memory_info)(a0, a1),
            Self::Hip(d) => (d.memory_info)(a0, a1),
        }
    }
    pub unsafe fn upload(&self, a0: DevicePtr, a1: *const c_void, a2: usize) -> c_int {
        match self {
            Self::Cuda(d) => (d.upload)(a0, a1, a2),
            Self::Hip(d) => (d.upload)(a0 as Handle, a1, a2),
        }
    }
    pub unsafe fn memset(&self, a0: DevicePtr, a1: u8, a2: usize) -> c_int {
        match self {
            Self::Cuda(d) => (d.memset)(a0, a1, a2),
            Self::Hip(d) => (d.memset)(a0 as Handle, a1 as c_int, a2),
        }
    }
    pub unsafe fn download(&self, a0: *mut c_void, a1: DevicePtr, a2: usize) -> c_int {
        match self {
            Self::Cuda(d) => (d.download)(a0, a1, a2),
            Self::Hip(d) => (d.download)(a0, a1 as Handle, a2),
        }
    }
    pub unsafe fn launch(
        &self,
        a0: Handle,
        a1: c_uint,
        a2: c_uint,
        a3: c_uint,
        a4: c_uint,
        a5: c_uint,
        a6: c_uint,
        a7: c_uint,
        a8: Handle,
        a9: *mut *mut c_void,
        a10: *mut *mut c_void,
    ) -> c_int {
        match self {
            Self::Cuda(d) => (d.launch)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10),
            Self::Hip(d) => (d.launch)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10),
        }
    }
    pub unsafe fn graph_create(&self, a0: *mut Handle, a1: c_uint) -> c_int {
        match self {
            Self::Cuda(d) => (d.graph_create)(a0, a1),
            Self::Hip(d) => (d.graph_create)(a0, a1),
        }
    }
    pub unsafe fn graph_destroy(&self, a0: Handle) -> c_int {
        match self {
            Self::Cuda(d) => (d.graph_destroy)(a0),
            Self::Hip(d) => (d.graph_destroy)(a0),
        }
    }
    pub unsafe fn graph_add_kernel(
        &self,
        a0: *mut Handle,
        a1: Handle,
        a2: *const Handle,
        a3: usize,
        a4: *const replay::KernelParams,
    ) -> c_int {
        match self {
            Self::Cuda(d) => (d.graph_add_kernel)(a0, a1, a2, a3, a4),
            Self::Hip(d) => {
                let p = &*a4;
                let hp = HipKernelParams {
                    block: [p.block_x, p.block_y, p.block_z],
                    extra: p.extra,
                    func: p.func,
                    grid: [p.grid_x, p.grid_y, p.grid_z],
                    arguments: p.arguments,
                    shared_bytes: p.shared_bytes,
                };
                (d.graph_add_kernel)(a0, a1, a2, a3, &hp)
            }
        }
    }
    pub unsafe fn graph_instantiate(
        &self,
        a0: *mut Handle,
        a1: Handle,
        a2: *mut Handle,
        a3: *mut c_char,
        a4: usize,
    ) -> c_int {
        match self {
            Self::Cuda(d) => (d.graph_instantiate)(a0, a1, a2, a3, a4),
            Self::Hip(d) => (d.graph_instantiate)(a0, a1, a2, a3, a4),
        }
    }
    pub unsafe fn graph_exec_destroy(&self, a0: Handle) -> c_int {
        match self {
            Self::Cuda(d) => (d.graph_exec_destroy)(a0),
            Self::Hip(d) => (d.graph_exec_destroy)(a0),
        }
    }
    pub unsafe fn graph_launch(&self, a0: Handle, a1: Handle) -> c_int {
        match self {
            Self::Cuda(d) => (d.graph_launch)(a0, a1),
            Self::Hip(d) => (d.graph_launch)(a0, a1),
        }
    }
    pub unsafe fn synchronize(&self) -> c_int {
        match self {
            Self::Cuda(d) => (d.synchronize)(),
            Self::Hip(d) => (d.synchronize)(),
        }
    }
    pub fn check(&self, code: c_int, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe {
            match self {
                Self::Cuda(d) => {
                    let mut message = std::ptr::null();
                    (d.error_string)(code, &mut message);
                    message
                }
                Self::Hip(d) => (d.error_string)(code),
            }
        };
        let detail = if message.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(message) }.to_string_lossy()
        };
        Err(Error(format!(
            "{} {operation}: {detail} (error {code})",
            self.backend().label()
        )))
    }
    pub fn architecture(&self, device: c_int) -> Result<String> {
        match self {
            Self::Cuda(d) => {
                let (mut major, mut minor) = (0, 0);
                self.check(
                    unsafe { (d.attribute)(&mut major, 75, device) },
                    "compute capability major",
                )?;
                self.check(
                    unsafe { (d.attribute)(&mut minor, 76, device) },
                    "compute capability minor",
                )?;
                Ok(format!("compute_{major}{minor}"))
            }
            Self::Hip(d) => {
                let mut properties: HipDeviceProperties = unsafe { std::mem::zeroed() };
                self.check(
                    unsafe { (d.properties)(&mut properties, device) },
                    "device properties (ROCm 6+)",
                )?;
                if properties.warpSize != 32 && properties.warpSize != 64 {
                    return Err(Error(format!(
                        "unsupported HIP wavefront size {}",
                        properties.warpSize
                    )));
                }
                let arch = &properties.gcnArchName;
                let end = arch
                    .iter()
                    .position(|&x| x == 0)
                    .ok_or_else(|| Error("unterminated HIP architecture".into()))?;
                let arch = std::str::from_utf8(&arch[..end]).map_err(|e| Error(e.to_string()))?;
                if !arch.starts_with("gfx")
                    || !arch
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b':' | b'+' | b'-'))
                {
                    return Err(Error(format!("invalid AMD HIP architecture {arch:?}")));
                }
                Ok(arch.into())
            }
        }
    }
    pub fn enter(&self, context: Handle) -> Result<Current<'_>> {
        self.check(unsafe { self.push(context) }, "push context")?;
        Ok(Current(self))
    }
}
#[repr(C)]
struct HipKernelParams {
    block: [c_uint; 3],
    extra: *mut *mut c_void,
    func: Handle,
    grid: [c_uint; 3],
    arguments: *mut *mut c_void,
    shared_bytes: c_uint,
}
// ROCm 6+ hipDeviceProp_tR0600, from AMD HIP hip_runtime_api.h.
// Keep the full versioned layout: the runtime writes all fields.
#[repr(C)]
#[allow(non_snake_case)]
struct HipDeviceProperties {
    name: [u8; 256],
    uuid: [u8; 16],
    luid: [u8; 8],
    luidDeviceNodeMask: c_uint,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: c_int,
    warpSize: c_int,
    memPitch: usize,
    maxThreadsPerBlock: c_int,
    maxThreadsDim: [c_int; 3],
    maxGridSize: [c_int; 3],
    clockRate: c_int,
    totalConstMem: usize,
    major: c_int,
    minor: c_int,
    textureAlignment: usize,
    texturePitchAlignment: usize,
    deviceOverlap: c_int,
    multiProcessorCount: c_int,
    kernelExecTimeoutEnabled: c_int,
    integrated: c_int,
    canMapHostMemory: c_int,
    computeMode: c_int,
    maxTexture1D: c_int,
    maxTexture1DMipmap: c_int,
    maxTexture1DLinear: c_int,
    maxTexture2D: [c_int; 2],
    maxTexture2DMipmap: [c_int; 2],
    maxTexture2DLinear: [c_int; 3],
    maxTexture2DGather: [c_int; 2],
    maxTexture3D: [c_int; 3],
    maxTexture3DAlt: [c_int; 3],
    maxTextureCubemap: c_int,
    maxTexture1DLayered: [c_int; 2],
    maxTexture2DLayered: [c_int; 3],
    maxTextureCubemapLayered: [c_int; 2],
    maxSurface1D: c_int,
    maxSurface2D: [c_int; 2],
    maxSurface3D: [c_int; 3],
    maxSurface1DLayered: [c_int; 2],
    maxSurface2DLayered: [c_int; 3],
    maxSurfaceCubemap: c_int,
    maxSurfaceCubemapLayered: [c_int; 2],
    surfaceAlignment: usize,
    concurrentKernels: c_int,
    ECCEnabled: c_int,
    pciBusID: c_int,
    pciDeviceID: c_int,
    pciDomainID: c_int,
    tccDriver: c_int,
    asyncEngineCount: c_int,
    unifiedAddressing: c_int,
    memoryClockRate: c_int,
    memoryBusWidth: c_int,
    l2CacheSize: c_int,
    persistingL2CacheMaxSize: c_int,
    maxThreadsPerMultiProcessor: c_int,
    streamPrioritiesSupported: c_int,
    globalL1CacheSupported: c_int,
    localL1CacheSupported: c_int,
    sharedMemPerMultiprocessor: usize,
    regsPerMultiprocessor: c_int,
    managedMemory: c_int,
    isMultiGpuBoard: c_int,
    multiGpuBoardGroupID: c_int,
    hostNativeAtomicSupported: c_int,
    singleToDoublePrecisionPerfRatio: c_int,
    pageableMemoryAccess: c_int,
    concurrentManagedAccess: c_int,
    computePreemptionSupported: c_int,
    canUseHostPointerForRegisteredMem: c_int,
    cooperativeLaunch: c_int,
    cooperativeMultiDeviceLaunch: c_int,
    sharedMemPerBlockOptin: usize,
    pageableMemoryAccessUsesHostPageTables: c_int,
    directManagedMemAccessFromHost: c_int,
    maxBlocksPerMultiProcessor: c_int,
    accessPolicyMaxWindowSize: c_int,
    reservedSharedMemPerBlock: usize,
    hostRegisterSupported: c_int,
    sparseHipArraySupported: c_int,
    hostRegisterReadOnlySupported: c_int,
    timelineSemaphoreInteropSupported: c_int,
    memoryPoolsSupported: c_int,
    gpuDirectRDMASupported: c_int,
    gpuDirectRDMAFlushWritesOptions: c_uint,
    gpuDirectRDMAWritesOrdering: c_int,
    memoryPoolSupportedHandleTypes: c_uint,
    deferredMappingHipArraySupported: c_int,
    ipcEventSupported: c_int,
    clusterLaunch: c_int,
    unifiedFunctionPointers: c_int,
    reserved: [c_int; 63],
    hipReserved: [c_int; 32],
    gcnArchName: [u8; 256],
    maxSharedMemoryPerMultiProcessor: usize,
    clockInstructionRate: c_int,
    arch: c_uint,
    hdpMemFlushCntl: *mut c_uint,
    hdpRegFlushCntl: *mut c_uint,
    cooperativeMultiDeviceUnmatchedFunc: c_int,
    cooperativeMultiDeviceUnmatchedGridDim: c_int,
    cooperativeMultiDeviceUnmatchedBlockDim: c_int,
    cooperativeMultiDeviceUnmatchedSharedMem: c_int,
    isLargeBar: c_int,
    asicRevision: c_int,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires ROCm runtime libraries, but no GPU"]
    fn hip_runtime_symbols_load_without_a_gpu() {
        assert_eq!(Driver::load(Backend::Hip).unwrap().backend(), Backend::Hip);
    }
    #[test]
    fn hip_versioned_abi_matches_amd_headers() {
        // hipDeviceProp_tR0600 and hipKernelNodeParams on 64-bit Linux.
        if cfg!(all(target_os = "linux", target_pointer_width = "64")) {
            assert_eq!(std::mem::size_of::<HipDeviceProperties>(), 1472);
            assert_eq!(std::mem::offset_of!(HipDeviceProperties, gcnArchName), 1160);
            assert_eq!(std::mem::offset_of!(HipDeviceProperties, warpSize), 308);
            assert_eq!(std::mem::size_of::<HipKernelParams>(), 64);
            assert_eq!(std::mem::offset_of!(HipKernelParams, func), 24);
            assert_eq!(std::mem::offset_of!(HipKernelParams, arguments), 48);
        }
    }
}
