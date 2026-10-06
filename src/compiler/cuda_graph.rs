//! GPU driver graph replay for one lowered executable. The legacy entry point
//! cuGraphAddKernelNode uses CUDA_KERNEL_NODE_PARAMS v1 (not the newer v2 ABI).
use super::*;

#[repr(C)]
pub(super) struct KernelParams {
    pub func: Handle,
    pub grid_x: c_uint,
    pub grid_y: c_uint,
    pub grid_z: c_uint,
    pub block_x: c_uint,
    pub block_y: c_uint,
    pub block_z: c_uint,
    pub shared_bytes: c_uint,
    pub arguments: *mut *mut c_void,
    pub extra: *mut *mut c_void,
}
pub(super) struct Arguments {
    // Keep the backing allocation alive while pointers into it are used.
    _values: Vec<DevicePtr>,
    pointers: Vec<*mut c_void>,
    grid: c_uint,
}
impl Arguments {
    pub fn new(
        k: &Kernel,
        inputs: &HashMap<usize, DevicePtr>,
        outputs: &[Memory],
        arena: DevicePtr,
        error: DevicePtr,
    ) -> Result<Self> {
        let mut values = k
            .bindings
            .iter()
            .map(|b| match b {
                Binding::Input(slot) => inputs[slot],
                Binding::Output(slot) => outputs[*slot].ptr,
                Binding::Arena(offset) => arena + *offset as u64,
            })
            .chain(std::iter::once(error))
            .collect::<Vec<_>>();
        let pointers = values
            .iter_mut()
            .map(|x| (x as *mut DevicePtr).cast())
            .collect();
        Ok(Self {
            _values: values,
            pointers,
            grid: c_uint::try_from(k.blocks)
                .map_err(|_| Error("CUDA launch exceeds grid limit".into()))?,
        })
    }
    pub fn params(&mut self, func: Handle) -> KernelParams {
        KernelParams {
            func,
            grid_x: self.grid,
            grid_y: 1,
            grid_z: 1,
            block_x: 256,
            block_y: 1,
            block_z: 1,
            shared_bytes: 0,
            arguments: self.pointers.as_mut_ptr(),
            extra: std::ptr::null_mut(),
        }
    }
}
struct Graph<'a> {
    driver: &'a Driver,
    handle: Handle,
}
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe {
            self.driver.graph_destroy(self.handle);
        }
    }
}
pub(super) struct Replay {
    context: Rc<Context>,
    handle: Handle,
    pub epoch: u64,
}
impl Replay {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        context: &Rc<Context>,
        kernels: &[Kernel],
        functions: &[Handle],
        inputs: &HashMap<usize, DevicePtr>,
        outputs: &[Memory],
        arena: DevicePtr,
        error: DevicePtr,
        epoch: u64,
    ) -> Result<Self> {
        let d = &context.driver;
        let mut handle = std::ptr::null_mut();
        d.check(unsafe { d.graph_create(&mut handle, 0) }, "create graph")?;
        let graph = Graph { driver: d, handle };
        let mut previous: Handle = std::ptr::null_mut();
        for (id, k) in kernels.iter().enumerate() {
            if k.elements == 0 {
                continue;
            }
            let mut args = Arguments::new(k, inputs, outputs, arena, error)?;
            let params = args.params(functions[id]);
            let mut node = std::ptr::null_mut();
            let deps = if previous.is_null() {
                std::ptr::null()
            } else {
                &previous
            };
            // The driver copies parameters and argument values during this call.
            // Preserve total order, including STORE/AFTER and reused arena regions.
            d.check(
                unsafe {
                    d.graph_add_kernel(
                        &mut node,
                        graph.handle,
                        deps,
                        usize::from(!previous.is_null()),
                        &params,
                    )
                },
                "add graph kernel",
            )?;
            previous = node;
        }
        let mut executable = std::ptr::null_mut();
        d.check(
            unsafe {
                d.graph_instantiate(
                    &mut executable,
                    graph.handle,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            },
            "instantiate graph",
        )?;
        Ok(Self {
            context: context.clone(),
            handle: executable,
            epoch,
        })
    }
    pub fn launch(&self) -> Result<()> {
        self.context.driver.check(
            unsafe {
                self.context
                    .driver
                    .graph_launch(self.handle, std::ptr::null_mut())
            },
            "launch graph",
        )
    }
}
impl Drop for Replay {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.driver.enter(self.context.handle) {
            unsafe {
                self.context.driver.graph_exec_destroy(self.handle);
            }
        }
    }
}
