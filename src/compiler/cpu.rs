//! C CPU backend. Views become index expressions; multiply/reduce contractions
//! lower to GEMM without materializing the broadcast product.
use super::pop::{numel, Arg, DType, Error, Graph, Op, ReduceOp, Result, Scalar, Value};
#[path = "cpu_build.rs"]
mod build;
#[path = "cuda_lower.rs"]
pub(crate) mod cuda_lower;
#[path = "cpu_gemm.rs"]
mod gemm;
#[path = "cpu_index.rs"]
mod index;
pub use build::{BuildInfo, BuildOptions, CpuTarget};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ffi::c_void,
    path::Path,
    rc::Rc,
    sync::Arc,
};
/// Persistent writable allocations shared by shape specializations. Model
/// semantics live in the graph; the runtime only owns zero-initialized bytes.
#[derive(Clone, Default)]
pub struct Runtime(Rc<RefCell<HashMap<usize, (DType, usize, Vec<u32>)>>>);
impl Runtime {
    pub fn reset_state(&self) -> Result<()> {
        let mut buffers = self
            .0
            .try_borrow_mut()
            .map_err(|_| Error("CPU state is executing".into()))?;
        for (_, _, words) in buffers.values_mut() {
            words.fill(0);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Tensor {
    F32(Arc<[f32]>),
    I32(Arc<[i32]>),
    U8(Arc<[u8]>),
    Bool(Arc<[u8]>),
}
impl Tensor {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::U8(_) => DType::U8,
            Self::Bool(_) => DType::Bool,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::F32(x) => x.len(),
            Self::I32(x) => x.len(),
            Self::Bool(x) | Self::U8(x) => x.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn ptr(&self) -> *const c_void {
        match self {
            Self::F32(x) => x.as_ptr().cast(),
            Self::I32(x) => x.as_ptr().cast(),
            Self::Bool(x) | Self::U8(x) => x.as_ptr().cast(),
        }
    }
    pub fn f32(&self) -> Result<&[f32]> {
        match self {
            Self::F32(x) => Ok(x),
            _ => Err(Error("expected f32 tensor".into())),
        }
    }
}
type Run = unsafe extern "C" fn(*const *const c_void, *const *mut c_void, usize) -> i32;
type ProfileRun =
    unsafe extern "C" fn(*const *const c_void, *const *mut c_void, usize, *mut u64, usize) -> i32;

pub struct ProfiledRun {
    pub outputs: Vec<Tensor>,
    pub counters: Vec<u64>,
}

pub struct Executable {
    _library: libloading::Library,
    run: Run,
    profile_run: Option<ProfileRun>,
    pub profile_metadata: Option<serde_json::Value>,
    pub cache_hit: bool,
    pub build_info: BuildInfo,
    inputs: Vec<(usize, DType, usize)>,
    state_slots: HashSet<usize>,
    runtime: Runtime,
    outputs: Vec<(DType, Vec<usize>)>,
    pub gemm_count: usize,
    pub source_path: std::path::PathBuf,
}
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
}
impl Executable {
    pub fn share_runtime(&mut self, runtime: &Runtime) {
        self.runtime = runtime.clone();
    }
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.run_with_threads(inputs, default_threads())
    }
    pub fn run_with_threads(&self, inputs: &[Tensor], threads: usize) -> Result<Vec<Tensor>> {
        self.run_internal(inputs, threads, false).map(|r| r.outputs)
    }
    pub fn run_profiled(&self, inputs: &[Tensor], threads: usize) -> Result<ProfiledRun> {
        if self.profile_run.is_none() {
            return Err(Error("executable was not compiled with profiling".into()));
        }
        self.run_internal(inputs, threads, true)
    }
    fn run_internal(
        &self,
        inputs: &[Tensor],
        threads: usize,
        profile: bool,
    ) -> Result<ProfiledRun> {
        if threads == 0 {
            return Err(Error("threads must be greater than zero".into()));
        }
        for &(slot, dtype, size) in &self.inputs {
            if self.state_slots.contains(&slot) {
                continue;
            }
            let x = inputs
                .get(slot)
                .ok_or_else(|| Error(format!("missing input slot {slot}")))?;
            if x.dtype() != dtype || x.len() != size {
                return Err(Error(format!(
                    "input slot {slot}: expected {dtype:?}[{size}], got {:?}[{}]",
                    x.dtype(),
                    x.len()
                )));
            }
        }
        let mut states = self
            .runtime
            .0
            .try_borrow_mut()
            .map_err(|_| Error("CPU state is executing".into()))?;
        let count = self
            .inputs
            .iter()
            .map(|(slot, _, _)| slot + 1)
            .max()
            .unwrap_or(0)
            .max(inputs.len());
        let mut ptrs = vec![std::ptr::null(); count];
        for (slot, input) in inputs.iter().enumerate() {
            ptrs[slot] = input.ptr();
        }
        for &(slot, dt, n) in &self.inputs {
            if !self.state_slots.contains(&slot) {
                continue;
            }
            let bytes = n
                .checked_mul(crate::compiler::cuda::dtype_bytes(dt))
                .ok_or_else(|| Error("state size overflow".into()))?;
            if bytes > 2 * 1024 * 1024 * 1024 {
                return Err(Error("state allocation exceeds 2 GiB".into()));
            }
            let entry = states
                .entry(slot)
                .or_insert_with(|| (dt, n, vec![0; bytes.div_ceil(4).max(1)]));
            if entry.0 != dt || entry.1 != n {
                *entry = (dt, n, vec![0; bytes.div_ceil(4).max(1)]);
            }
            ptrs[slot] = entry.2.as_mut_ptr().cast();
        }
        // Own mutable buffers separately: no aliasing through Arc during execution.
        enum Buffer {
            F(Vec<f32>),
            I(Vec<i32>),
            B(Vec<u8>),
            U(Vec<u8>),
        }
        let mut buffers = Vec::new();
        for (dt, shape) in &self.outputs {
            let n = numel(shape)?;
            buffers.push(match dt {
                DType::F32 => Buffer::F(vec![0.; n]),
                DType::I32 => Buffer::I(vec![0; n]),
                DType::Bool => Buffer::B(vec![0; n]),
                DType::U8 => Buffer::U(vec![0; n]),
                _ => {
                    return Err(Error(
                        "CPU outputs require concrete dtypes; add cast".into(),
                    ))
                }
            });
        }
        let outptrs: Vec<*mut c_void> = buffers
            .iter_mut()
            .map(|x| match x {
                Buffer::F(v) => v.as_mut_ptr().cast(),
                Buffer::I(v) => v.as_mut_ptr().cast(),
                Buffer::B(v) | Buffer::U(v) => v.as_mut_ptr().cast(),
            })
            .collect();
        let mut counters = if profile {
            vec![
                0;
                self.profile_metadata.as_ref().unwrap()["stats_length"]
                    .as_u64()
                    .unwrap() as usize
            ]
        } else {
            Vec::new()
        };
        let status = unsafe {
            if profile {
                (self.profile_run.unwrap())(
                    ptrs.as_ptr(),
                    outptrs.as_ptr(),
                    threads,
                    counters.as_mut_ptr(),
                    counters.len(),
                )
            } else {
                (self.run)(ptrs.as_ptr(), outptrs.as_ptr(), threads)
            }
        };
        if status != 0 {
            return Err(Error(
                match status {
                    1 => "CPU workspace allocation failed",
                    2 => "INDEX out of bounds",
                    3 => "CPU worker pool initialization failed or invalid thread count",
                    _ => "CPU execution failed",
                }
                .into(),
            ));
        }
        let outputs = buffers
            .into_iter()
            .map(|x| match x {
                Buffer::F(v) => Tensor::F32(v.into()),
                Buffer::I(v) => Tensor::I32(v.into()),
                Buffer::B(v) => Tensor::Bool(v.into()),
                Buffer::U(v) => Tensor::U8(v.into()),
            })
            .collect();
        Ok(ProfiledRun { outputs, counters })
    }
}
fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::WeakFloat => "double",
        DType::I32 => "int32_t",
        DType::WeakInt => "int64_t",
        DType::Bool | DType::U8 => "uint8_t",
        DType::Void => "uint8_t",
    }
}
fn coord(index: &str, shape: &[usize], axis: usize) -> String {
    if shape[axis] <= 1 {
        return "0".into();
    }
    let stride: usize = shape[axis + 1..].iter().product();
    let term = if stride <= 1 {
        format!("({index})")
    } else {
        format!("(({index})/{stride})")
    };
    if shape[..axis].iter().all(|&d| d == 1) {
        term
    } else {
        format!("({term}%{})", shape[axis])
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .map(|i| shape[i + 1..].iter().product())
        .collect()
}
struct Allocation {
    name: String,
    bytes: usize,
    first: usize,
    last: usize,
    offset: usize,
}
fn pointwise(op: Op) -> bool {
    matches!(
        op,
        Op::Cast
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Fdiv
            | Op::Max
            | Op::Cmplt
            | Op::Neg
            | Op::Where
            | Op::Exp2
            | Op::Log2
            | Op::Sqrt
            | Op::Sin
    )
}
struct Emitter<'a> {
    g: &'a Graph,
    names: HashMap<Value, String>,
    material: HashSet<Value>,
    inline_values: HashSet<Value>,
    overrides: HashMap<Value, String>,
    allocations: Vec<Allocation>,
    direct_outputs: HashMap<String, usize>,
    decl: String,
    body: String,
    peak_workspace: usize,
    peak_packed_workspace: usize,
    parallel_tasks: usize,
    kernels: String,
    kernel_count: usize,
    types: HashMap<String, DType>,
    profile: bool,
    wrap_signed_arithmetic: bool,
    labels: HashMap<Value, Vec<serde_json::Value>>,
    kernel_metadata: Vec<serde_json::Value>,
}
impl Emitter<'_> {
    fn store_parts(&self, v: Value, value: &str, cuda: bool) -> Result<(usize, String)> {
        let node = self.g.node(v)?;
        let dest = self.g.node(node.src()[0])?;
        let shape = self.shape(dest.src()[0]);
        let row = numel(&shape[1..])?;
        let ix = self.read(dest.src()[1], &format!("i/{}", row.max(1)))?;
        let base = self.g.writable_base(dest.src()[0])?;
        let target = &self.names[&base];
        let error = if cuda {
            "atomicExch(error,2);return;"
        } else {
            "err=2;goto cleanup;"
        };
        let body = format!(
            "int64_t ix={ix}; if(ix<0 || ix>={}) {{{error}}} {target}[ix*{row}+i%{}]={value};",
            shape[0],
            row.max(1)
        );
        Ok((numel(&self.shape(node.src()[1]))?, body))
    }
    fn store_snapshot_needed(&self, mut value: Value) -> bool {
        loop {
            if self.material.contains(&value) {
                return false;
            }
            let node = self.g.node(value).unwrap();
            match node.op() {
                Op::Reshape
                | Op::Expand
                | Op::Permute
                | Op::Shrink
                | Op::Flip
                | Op::Load
                | Op::After => value = node.src()[0],
                Op::Param => return matches!(node.arg(),Arg::Param(p) if p.writable),
                _ => return false,
            }
        }
    }
    fn shape(&self, v: Value) -> Vec<usize> {
        self.g.node(v).unwrap().shape().unwrap().to_vec()
    }
    fn alloc(&mut self, name: &str, dt: DType, n: usize) -> Result<()> {
        let bytes = n
            .checked_mul(match dt {
                DType::WeakFloat | DType::WeakInt => 8,
                DType::F32 | DType::I32 => 4,
                _ => 1,
            })
            .ok_or_else(|| Error("CPU workspace overflow".into()))?;
        if bytes > 2 * 1024 * 1024 * 1024 {
            return Err(Error("CPU workspace exceeds 2 GiB".into()));
        }
        self.types.insert(name.into(), dt);
        let ct = ctype(dt);
        self.decl += &format!("{ct} *{name}=NULL;\n");
        if let Some(slot) = self.direct_outputs.get(name) {
            self.body += &format!("{name}=({ct}*)outputs[{slot}];\n");
        } else {
            self.allocations.push(Allocation {
                name: name.into(),
                bytes,
                first: self.kernel_count,
                last: self.kernel_count,
                offset: 0,
            });
            self.body += &format!("{name}=({ct}*)(arena+PUP_OFFSET_{name});\n");
        }
        Ok(())
    }
    fn mark_reads(&mut self, code: &str, step: usize) {
        let words: HashSet<_> = code
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .collect();
        for a in &mut self.allocations {
            if words.contains(a.name.as_str()) {
                a.last = a.last.max(step);
            }
        }
    }
    fn plan_memory(&mut self) -> Result<serde_json::Value> {
        // Largest allocations first, with closed lifetime intervals: a producer
        // may not overwrite an input that is still being read in the same kernel.
        let mut ids: Vec<_> = (0..self.allocations.len()).collect();
        ids.sort_by_key(|&i| (std::cmp::Reverse(self.allocations[i].bytes), i));
        let mut placed: Vec<usize> = vec![];
        for id in ids {
            let a = &self.allocations[id];
            let mut occupied: Vec<_> = placed
                .iter()
                .filter_map(|&j| {
                    let b = &self.allocations[j];
                    (a.bytes > 0 && b.bytes > 0 && a.first <= b.last && b.first <= a.last)
                        .then_some((b.offset, b.offset + b.bytes))
                })
                .collect();
            occupied.sort_unstable();
            let mut offset = 0usize;
            for (begin, end) in occupied {
                if offset.checked_add(a.bytes).is_some_and(|x| x <= begin) {
                    break;
                }
                if offset < end {
                    offset = end
                        .checked_add(7)
                        .ok_or_else(|| Error("CPU workspace overflow".into()))?
                        & !7;
                }
            }
            let end = offset
                .checked_add(a.bytes)
                .ok_or_else(|| Error("CPU workspace overflow".into()))?;
            if end > 2 * 1024 * 1024 * 1024 {
                return Err(Error("CPU workspace exceeds 2 GiB".into()));
            }
            self.peak_workspace = self.peak_workspace.max(end);
            self.allocations[id].offset = offset;
            placed.push(id);
        }
        let mut definitions = String::new();
        for a in &self.allocations {
            definitions += &format!("const size_t PUP_OFFSET_{}={}ULL;\n", a.name, a.offset);
        }
        self.decl += &definitions;
        let peak_live_bytes = (0..=self.kernel_count)
            .map(|step| {
                self.allocations
                    .iter()
                    .filter(|a| a.first <= step && step <= a.last)
                    .map(|a| a.bytes)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        Ok(
            serde_json::json!({"peak_live_bytes":peak_live_bytes, "allocations":self.allocations.iter().map(|a|
            serde_json::json!({"value":a.name,"bytes":a.bytes,"first_kernel":a.first,"last_kernel":a.last,"offset":a.offset})).collect::<Vec<_>>() }),
        )
    }
    fn kernel(
        &mut self,
        value: Value,
        start: usize,
        packed_bytes: usize,
        flops: usize,
        phase_timing_kind: &str,
    ) {
        let mut code = self.body.split_off(start);
        let mut vars: Vec<String> = code
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|word| self.types.contains_key(*word))
            .map(str::to_string)
            .collect();
        vars.sort();
        vars.dedup();
        let params = vars
            .iter()
            .map(|v| format!("{} *{}", ctype(self.types[v]), v))
            .collect::<Vec<_>>();
        // Destinations are writable; sources are not exposed outside this module.
        let params = params.join(", ");
        let args = vars.join(", ");
        self.mark_reads(&code, self.kernel_count);
        let id = self.kernel_count;
        self.kernel_count += 1;
        let n = self
            .g
            .node(value)
            .unwrap()
            .shape()
            .map(|s| numel(s).unwrap())
            .unwrap_or(1);
        let loop_head = format!("for(size_t i=0;i<{n};i++)");
        if flops == 0
            && n >= 32768
            && code.starts_with(&loop_head)
            && !code.contains("goto cleanup;")
        {
            self.parallel_tasks = self.parallel_tasks.max(n.div_ceil(4096));
            let fields = vars
                .iter()
                .map(|v| format!("{} *{v};", ctype(self.types[v])))
                .collect::<Vec<_>>()
                .join("\n");
            let locals = vars
                .iter()
                .map(|v| format!("{} *restrict {v}=ctx->{v};", ctype(self.types[v])))
                .collect::<Vec<_>>()
                .join("\n");
            let worker = code.replacen(&loop_head, "for(size_t i=begin;i<end;i++)", 1);
            self.kernels += &format!("typedef struct {{ {fields} }} point_context{id};\nstatic void point_tile{id}(const pup_pool *pool,size_t id) {{ const point_context{id} *ctx=pool->context;\n{locals}\nconst size_t base={n}/pool->threads,extra={n}%pool->threads;\nconst size_t begin=id*base+(id<extra?id:extra),end=begin+base+(id<extra);\n{worker}}}\n");
            code = format!(
                "point_context{id} ctx={{{args}}};\npup_dispatch(pool,point_tile{id},&ctx);\n"
            );
        }
        let profile_param = if self.profile {
            ", uint64_t *stats"
        } else {
            ""
        };
        self.kernels+=&format!("static __attribute__((noinline)) int kernel{id}(pup_pool *pool{profile_param}{}{}) {{int err=0;\n{}return 0;}}\n",if params.is_empty(){""}else{", "},params,code.replace("goto cleanup;","return err;"));
        if self.profile {
            let offset = 4 + id * 4;
            let node = self.g.node(value).unwrap();
            self.kernel_metadata.push(serde_json::json!({
                "id": id, "value": self.names[&value], "op": if flops > 0 { "Matmul".into() } else { format!("{:?}", node.op()) },
                "dtype": node.dtype(), "shape": node.shape(), "bindings": self.labels.get(&value).cloned().unwrap_or_default(),
                "packed_bytes": packed_bytes, "matmul_flops": flops, "stats_offset": offset,
                "phase_timing_kind": phase_timing_kind
            }));
            self.body += &format!("{{ uint64_t started=pup_clock_ns();\nerr=kernel{id}(&pool,stats{}{});\nstats[{offset}]++;stats[{}]+=pup_clock_ns()-started;\nif(err) goto cleanup; }}\n",if args.is_empty(){""}else{", "},args,offset+1);
        } else {
            self.body += &format!(
                "if((err=kernel{id}(&pool{}{}))) goto cleanup;\n",
                if args.is_empty() { "" } else { ", " },
                args
            );
        }
    }

    fn read(&self, v: Value, i: &str) -> Result<String> {
        let node = self.g.node(v)?;
        let src = node.src();
        let shape = self.shape(v);
        if let Some(expr) = self.overrides.get(&v) {
            // CUDA's literal-range expressions need the consumer's view index;
            // scalar accumulator overrides do not contain this placeholder.
            return Ok(expr.replace("$index", i));
        }
        if self.material.contains(&v) || node.op() == Op::Param {
            return Ok(format!("{}[{i}]", self.names[&v]));
        }
        match node.op() {
            Op::Const => {
                if let Arg::Scalar(s) = node.arg() {
                    Ok(match s {
                        Scalar::Int(n) => format!("INT64_C({n})"),
                        Scalar::Bool(b) => {
                            if *b {
                                "1".into()
                            } else {
                                "0".into()
                            }
                        }
                        Scalar::Float(bits) => {
                            let f = f64::from_bits(*bits);
                            if f.is_nan() {
                                "NAN".into()
                            } else if f == f64::INFINITY {
                                "INFINITY".into()
                            } else if f == f64::NEG_INFINITY {
                                "(-INFINITY)".into()
                            } else {
                                format!("({f:.17e})")
                            }
                        }
                    })
                } else {
                    unreachable!()
                }
            }
            Op::Reshape | Op::Load | Op::After => self.read(src[0], i),
            Op::Stack if self.inline_values.contains(&v) => {
                let part = numel(&self.shape(src[0]))?.max(1);
                let mut expression = self.read(*src.last().unwrap(), &format!("(({i})%{part})"))?;
                for (j, &s) in src[..src.len() - 1].iter().enumerate().rev() {
                    let x = self.read(s, &format!("(({i})%{part})"))?;
                    expression = format!("(({i})<{}?({x}):({expression}))", (j + 1) * part);
                }
                Ok(expression)
            }
            Op::Window => {
                let coords = (0..shape.len())
                    .map(|axis| coord(i, &shape, axis))
                    .collect::<Vec<_>>();
                self.read_coordinates(v, &coords)
            }
            Op::Expand => self.read(
                src[0],
                &format!("(({i})%{})", numel(&self.shape(src[0]))?.max(1)),
            ),
            Op::Permute if matches!(node.arg(), Arg::Axes(axes) if axes.iter().copied().eq(0..axes.len())) => {
                self.read(src[0], i)
            }
            Op::Permute | Op::Flip | Op::Shrink => {
                let coords = (0..shape.len())
                    .map(|axis| coord(i, &shape, axis))
                    .collect::<Vec<_>>();
                self.read_coordinates(v, &coords)
            }
            op if pointwise(op) => self.elementwise(v, i),
            _ => Err(Error(format!("missing CPU storage for {:?}", node.op()))),
        }
    }
    // Keep contraction coordinates through views instead of flattening and then
    // dividing them back out. Besides simpler C, this lets the C compiler see
    // contiguous operand loads without proving bounds on nested tile loops.
    fn read_coordinates(&self, v: Value, coords: &[String]) -> Result<String> {
        let node = self.g.node(v)?;
        let shape = self.shape(v);
        let src = node.src();
        let flat = coords
            .iter()
            .zip(strides(&shape))
            .filter_map(|(c, stride)| {
                (c != "0").then(|| {
                    if stride == 1 {
                        format!("({c})")
                    } else {
                        format!("({c})*{stride}")
                    }
                })
            })
            .collect::<Vec<_>>()
            .join("+");
        let flat = if flat.is_empty() { "0" } else { &flat };
        if self.material.contains(&v) || node.op() == Op::Param || self.overrides.contains_key(&v) {
            return self.read(v, flat);
        }
        match node.op() {
            Op::Window => {
                let step = self.dims(src[2])?;
                self.read_coordinates(
                    src[0],
                    &[
                        coords[0].clone(),
                        coords[1].clone(),
                        format!("({})*{}+({})", coords[2], step[0], coords[4]),
                        format!("({})*{}+({})", coords[3], step[1], coords[5]),
                    ],
                )
            }
            Op::Permute => {
                let Arg::Axes(axes) = node.arg() else {
                    unreachable!()
                };
                let mut old = vec!["0".to_owned(); axes.len()];
                for (j, &axis) in axes.iter().enumerate() {
                    old[axis] = coords[j].clone();
                }
                self.read_coordinates(src[0], &old)
            }
            Op::Reshape => {
                let old = self.shape(src[0]);
                let old_coords = index::reshape_coordinates(&old, &shape, coords)?;
                self.read_coordinates(src[0], &old_coords)
            }
            Op::Expand => {
                let rank = self.shape(src[0]).len();
                self.read_coordinates(src[0], &coords[coords.len() - rank..])
            }
            Op::Load | Op::After => self.read_coordinates(src[0], coords),
            Op::Flip => {
                let Arg::Flip(flags) = node.arg() else {
                    unreachable!()
                };
                let old = coords
                    .iter()
                    .zip(flags)
                    .zip(&shape)
                    .map(|((c, &flip), &d)| {
                        if flip {
                            format!("{}-({c})", d.saturating_sub(1))
                        } else {
                            c.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                self.read_coordinates(src[0], &old)
            }
            Op::Shrink => {
                let offsets = self.dims(src[1])?;
                let old = coords
                    .iter()
                    .zip(offsets)
                    .map(|(c, off)| format!("({c})+{off}"))
                    .collect::<Vec<_>>();
                self.read_coordinates(src[0], &old)
            }
            op if pointwise(op) => {
                let xs = src
                    .iter()
                    .map(|&s| {
                        let child_shape = self.shape(s);
                        let offset = shape.len() - child_shape.len();
                        let child_coords = child_shape
                            .iter()
                            .enumerate()
                            .map(|(j, &d)| {
                                if d == 1 {
                                    "0".into()
                                } else {
                                    coords[offset + j].clone()
                                }
                            })
                            .collect::<Vec<_>>();
                        self.read_coordinates(s, &child_coords)
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.elementwise_expression(v, xs)
            }
            _ => self.read(v, flat),
        }
    }
    // Layout analysis selects packing; generated operand reads remain general.
    fn storage_strides(&self, v: Value) -> Option<Vec<isize>> {
        let node = self.g.node(v).ok()?;
        let shape = self.shape(v);
        if self.material.contains(&v) || node.op() == Op::Param {
            return Some(strides(&shape).into_iter().map(|s| s as isize).collect());
        }
        let src = node.src();
        let old = self.storage_strides(*src.first()?)?;
        match node.op() {
            Op::Load | Op::Shrink | Op::After => Some(old),
            Op::Reshape => index::reshape_strides(&self.shape(src[0]), &shape, &old),
            Op::Permute => {
                let Arg::Axes(axes) = node.arg() else {
                    return None;
                };
                Some(axes.iter().map(|&axis| old[axis]).collect())
            }
            Op::Expand => Some(
                vec![0; shape.len() - old.len()]
                    .into_iter()
                    .chain(old)
                    .collect(),
            ),
            Op::Flip => {
                let Arg::Flip(flags) = node.arg() else {
                    return None;
                };
                Some(
                    old.iter()
                        .zip(flags)
                        .map(|(&s, &flip)| if flip { -s } else { s })
                        .collect(),
                )
            }
            Op::Window => {
                let step = self.dims(src[2]).ok()?;
                Some(vec![
                    old[0],
                    old[1],
                    old[2] * step[0] as isize,
                    old[3] * step[1] as isize,
                    old[2],
                    old[3],
                ])
            }
            _ => None,
        }
    }
    fn elementwise(&self, v: Value, i: &str) -> Result<String> {
        let node = self.g.node(v)?;
        let src = node.src();
        let shape = self.shape(v);
        let xs = src
            .iter()
            .map(|&s| self.broadcast(s, &shape, i))
            .collect::<Result<Vec<_>>>()?;
        self.elementwise_expression(v, xs)
    }
    fn elementwise_expression(&self, v: Value, xs: Vec<String>) -> Result<String> {
        let node = self.g.node(v)?;
        let src = node.src();
        let dt = node.dtype();
        let scalar_type = if node.op() == Op::Cmplt {
            src.iter()
                .map(|&v| self.g.node(v).unwrap().dtype())
                .find(|d| !d.is_weak())
        } else {
            Some(dt)
        };
        let xs = src
            .iter()
            .zip(xs)
            .map(|(&s, x)| {
                if self.g.node(s).unwrap().dtype().is_weak()
                    && scalar_type.is_some_and(|d| matches!(d, DType::F32 | DType::I32 | DType::U8))
                {
                    format!("({})({x})", ctype(scalar_type.unwrap()))
                } else {
                    x
                }
            })
            .collect::<Vec<_>>();
        let x = &xs[0];
        let y = xs.get(1).map(String::as_str).unwrap_or("");
        if self.wrap_signed_arithmetic && matches!(dt, DType::I32 | DType::WeakInt) {
            let unsigned = if dt == DType::I32 {
                "unsigned int"
            } else {
                "unsigned long long"
            };
            let operation = match node.op() {
                Op::Add => Some(format!("({unsigned})({x})+({unsigned})({y})")),
                Op::Sub => Some(format!("({unsigned})({x})-({unsigned})({y})")),
                Op::Mul => Some(format!("({unsigned})({x})*({unsigned})({y})")),
                Op::Neg => Some(format!("({unsigned})0-({unsigned})({x})")),
                _ => None,
            };
            if let Some(operation) = operation {
                return Ok(format!("({})({operation})", ctype(dt)));
            }
        }
        let expr = match node.op() {
            Op::Cast if dt == DType::Bool => format!("({x})!=0"),
            Op::Cast => format!("({})({x})", ctype(dt)),
            Op::Add => format!("({x})+({y})"),
            Op::Sub => format!("({x})-({y})"),
            Op::Mul => format!("({x})*({y})"),
            Op::Fdiv => format!("({x})/({y})"),
            Op::Max => format!("({x})>({y})?({x}):({y})"),
            Op::Cmplt => format!("({x})<({y})"),
            Op::Neg => format!("-({x})"),
            Op::Where => format!("({x})?({y}):({})", xs[2]),
            Op::Exp2 => format!("{}({x})", if dt == DType::F32 { "exp2f" } else { "exp2" }),
            Op::Log2 => format!("{}({x})", if dt == DType::F32 { "log2f" } else { "log2" }),
            Op::Sqrt => format!("{}({x})", if dt == DType::F32 { "sqrtf" } else { "sqrt" }),
            Op::Sin => format!("{}({x})", if dt == DType::F32 { "sinf" } else { "sin" }),
            op => return Err(Error(format!("CPU lowering missing {op:?}"))),
        };
        Ok(format!("({})({expr})", ctype(dt)))
    }
    fn dims(&self, v: Value) -> Result<Vec<usize>> {
        let n = self.g.node(v)?;
        let vals = if n.op() == Op::Stack {
            n.src().to_vec()
        } else {
            vec![v]
        };
        vals.iter()
            .map(|&v| match self.g.node(v)?.arg() {
                Arg::Scalar(Scalar::Int(n)) => Ok(*n as usize),
                _ => Err(Error("expected static dimensions".into())),
            })
            .collect()
    }
    fn broadcast(&self, v: Value, target: &[usize], i: &str) -> Result<String> {
        let shape = self.shape(v);
        if shape == target {
            return self.read(v, i);
        }
        let st = strides(&shape);
        let terms: Vec<_> = shape
            .iter()
            .enumerate()
            .filter(|(_, d)| **d != 1)
            .map(|(j, _)| {
                format!(
                    "{}*{}",
                    coord(i, target, j + target.len() - shape.len()),
                    st[j]
                )
            })
            .collect();
        self.read(
            v,
            &if terms.is_empty() {
                "0".into()
            } else {
                format!("({})", terms.join("+"))
            },
        )
    }
}

/// Returns generated C and the number of fused multiply/reduce contractions.
pub fn emit(g: &Graph, root: Value) -> Result<(String, usize)> {
    emit_impl(g, root, None)
}

/// Opt-in profiling ABI; ordinary emission retains no timing instrumentation.
pub fn emit_profiled(program: &super::source::Program) -> Result<(String, usize)> {
    emit_impl(&program.graph, program.root, Some(&program.bindings))
}

fn emit_impl(
    g: &Graph,
    root: Value,
    bindings: Option<&[super::source::Binding]>,
) -> Result<(String, usize)> {
    let mut labels = HashMap::<Value, Vec<serde_json::Value>>::new();
    for binding in bindings.unwrap_or_default() {
        labels
            .entry(binding.value)
            .or_default()
            .push(serde_json::json!({"name": binding.name, "line": binding.line}));
    }
    let outputs = if g.node(root)?.op() == Op::Sink {
        g.node(root)?.src().to_vec()
    } else {
        vec![root]
    };
    let mut output_order = outputs.clone();
    if outputs
        .iter()
        .any(|&v| g.node(v).unwrap().shape().is_none())
    {
        return Err(Error(
            "outputs must be tensor values; attach writes through AFTER".into(),
        ));
    }
    if !g
        .toposort(root)?
        .iter()
        .any(|&v| g.node(v).unwrap().op() == Op::Store)
    {
        output_order
            .sort_by_key(|&v| numel(g.node(v).unwrap().shape().unwrap()).unwrap_or(usize::MAX));
    }
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for &out in &output_order {
        for v in g.toposort(out)? {
            if seen.insert(v) {
                order.push(v);
            }
        }
    }

    let mut uses = HashMap::<Value, usize>::new();
    for &v in &order {
        for &s in g.node(v)?.src() {
            *uses.entry(s).or_default() += 1;
        }
    }
    for &out in &outputs {
        *uses.entry(out).or_default() += 1;
    }
    // STORE addresses do not require a gather of the destination's old data.
    let mut store_targets = HashSet::new();
    let mut store_values = HashSet::new();
    for &v in &order {
        if g.node(v)?.op() == Op::Store {
            store_targets.insert(g.node(v)?.src()[0]);
            store_values.insert(g.node(v)?.src()[1]);
        }
    }
    let address_only = store_targets
        .into_iter()
        .filter(|&target| {
            !outputs.contains(&target)
                && order.iter().all(|&v| {
                    !g.node(v).unwrap().src().contains(&target)
                        || (g.node(v).unwrap().op() == Op::Store
                            && g.node(v).unwrap().src()[0] == target
                            && g.node(v).unwrap().src()[1] != target)
                })
        })
        .collect::<HashSet<_>>();
    // A store payload may be a view of pointwise work. Materialize its
    // underlying value before the write so overlap cannot change later reads.
    for mut value in store_values.clone() {
        loop {
            let node = g.node(value)?;
            if matches!(
                node.op(),
                Op::Reshape
                    | Op::Expand
                    | Op::Permute
                    | Op::Flip
                    | Op::Shrink
                    | Op::Load
                    | Op::After
                    | Op::Window
            ) {
                value = node.src()[0];
                store_values.insert(value);
            } else {
                break;
            }
        }
    }
    let mut fused = HashMap::new();
    let mut skip = HashSet::new();
    skip.extend(address_only);
    for &v in &order {
        let n = g.node(v)?;
        if n.arg()
            != &(Arg::Reduce {
                op: ReduceOp::Add,
                num_axes: 1,
            })
            || n.dtype() != DType::F32
        {
            continue;
        }
        let p = n.src()[0];
        let pn = g.node(p)?;
        if pn.op() != Op::Permute || uses[&p] != 1 {
            continue;
        }
        let mul = pn.src()[0];
        let mn = g.node(mul)?;
        if mn.op() != Op::Mul || uses[&mul] != 1 {
            continue;
        }
        let shape = mn.shape().unwrap();
        let rank = shape.len();
        if !(rank == 3 || rank == 4) {
            continue;
        }
        let axes: Vec<_> = std::iter::once(rank - 1).chain(0..rank - 1).collect();
        if pn.arg() != &Arg::Axes(axes) {
            continue;
        }
        let a = g.node(mn.src()[0])?;
        let b = g.node(mn.src()[1])?;
        if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
            continue;
        }
        let (ash, bsh) = (a.shape().unwrap(), b.shape().unwrap());
        if ash.len() != rank
            || bsh.len() != rank
            || ash[rank - 2] != 1
            || bsh[rank - 3] != 1
            || ash[rank - 1] != bsh[rank - 1]
        {
            continue;
        }
        fused.insert(v, (mn.src()[0], mn.src()[1], shape.to_vec()));
        skip.insert(p);
        skip.insert(mul);
    }
    let mut consumers = HashMap::<Value, Vec<Value>>::new();
    for &v in &order {
        for &input in g.node(v)?.src() {
            consumers.entry(input).or_default().push(v);
        }
    }
    // Shape operands are compile-time metadata, not executable tensor work.
    for &v in &order {
        if g.node(v)?.op() == Op::Stack
            && !outputs.contains(&v)
            && !store_values.contains(&v)
            && consumers.get(&v).is_some_and(|cs| {
                cs.iter().all(|&c| {
                    let n = g.node(c).unwrap();
                    matches!(
                        n.op(),
                        Op::Reshape
                            | Op::Expand
                            | Op::Permute
                            | Op::Shrink
                            | Op::Flip
                            | Op::Pad
                            | Op::Window
                    ) && n.src()[0] != v
                })
            })
        {
            skip.insert(v);
        }
    }
    let mut contractions = HashMap::new();
    let mut reductions = HashMap::new();
    let mut claimed = HashSet::new();
    for &start in &order {
        if g.node(start)?.op() != Op::Reduce || g.node(start)?.dtype() != DType::F32 {
            continue;
        }
        let mut end = start;
        for _ in 0..8 {
            if outputs.contains(&end) {
                break;
            }
            let Some(next) = consumers
                .get(&end)
                .filter(|xs| xs.len() == 1)
                .map(|xs| xs[0])
            else {
                break;
            };
            let node = g.node(next)?;
            if claimed.contains(&next)
                || !pointwise(node.op())
                || node.dtype() != DType::F32
                || node.shape() != g.node(start)?.shape()
            {
                break;
            }
            if store_values.contains(&end) {
                break;
            }
            claimed.insert(next);
            skip.insert(end);
            end = next;
        }
        if let Some((a, b, product)) = fused.get(&start) {
            contractions.insert(end, (start, *a, *b, product.clone()));
        } else {
            reductions.insert(end, start);
        }
    }
    let mut costs = HashMap::<Value, usize>::new();
    let mut inline_values = HashSet::new();
    for &v in &order {
        let node = g.node(v)?;
        let cost = 1 + node
            .src()
            .iter()
            .map(|s| costs.get(s).copied().unwrap_or(0))
            .sum::<usize>()
            .min(64);
        let expensive = matches!(node.op(), Op::Exp2 | Op::Log2 | Op::Sqrt | Op::Sin);
        let shared_div = node.op() == Op::Fdiv && numel(node.shape().unwrap())? <= 8192;
        if pointwise(node.op())
            && cost <= 24
            && (!expensive && !shared_div || uses.get(&v).copied().unwrap_or(0) <= 1)
            && !outputs.contains(&v)
            && !skip.contains(&v)
            && !store_values.contains(&v)
            && !contractions.contains_key(&v)
            && !reductions.contains_key(&v)
        {
            inline_values.insert(v);
            costs.insert(v, cost);
        } else if matches!(
            node.op(),
            Op::Reshape | Op::Expand | Op::Permute | Op::Flip | Op::Shrink | Op::Load | Op::After
        ) {
            costs.insert(v, cost);
        }
    }
    let mut e = Emitter {
        g,
        names: order
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, format!("v{i}")))
            .collect(),
        material: HashSet::new(),
        inline_values,
        overrides: HashMap::new(),
        allocations: vec![],
        direct_outputs: HashMap::new(),
        decl: String::new(),
        body: String::new(),
        peak_workspace: 0,
        peak_packed_workspace: 0,
        parallel_tasks: 1,
        kernels: String::new(),
        kernel_count: 0,
        types: HashMap::new(),
        profile: bindings.is_some(),
        wrap_signed_arithmetic: false,
        labels,
        kernel_metadata: Vec::new(),
    };
    for (slot, &out) in outputs.iter().enumerate() {
        if !matches!(
            g.node(out)?.op(),
            Op::Const
                | Op::Param
                | Op::Reshape
                | Op::Expand
                | Op::Permute
                | Op::Shrink
                | Op::Flip
                | Op::Load
                | Op::After
                | Op::Window
        ) {
            e.direct_outputs
                .entry(e.names[&out].clone())
                .or_insert(slot);
        }
    }
    for &v in &order {
        let node = g.node(v)?;
        let name = e.names[&v].clone();
        let src = node.src();
        let dt = node.dtype();
        if skip.contains(&v) || e.inline_values.contains(&v) || node.op() == Op::Sink {
            continue;
        }
        if node.op() == Op::Store {
            let payload = src[1];
            let n = numel(&e.shape(payload))?;
            let value = if e.store_snapshot_needed(payload) {
                let snapshot = format!("state_snapshot{}", e.kernel_count);
                let value = e.read(payload, "i")?;
                e.alloc(&snapshot, g.node(payload)?.dtype(), n)?;
                let begin = e.body.len();
                e.body += &format!("for(size_t i=0;i<{n};i++) {snapshot}[i]={value};\n");
                e.kernel(v, begin, 0, 0, "wall");
                format!("{snapshot}[i]")
            } else {
                e.read(payload, "i")?
            };
            let (_, body) = e.store_parts(v, &value, false)?;
            let begin = e.body.len();
            e.body += &format!("for(size_t i=0;i<{n};i++) {{{body}}}\n");
            e.kernel(v, begin, 0, 0, "wall");
            continue;
        }
        let shape = e.shape(v);
        let n = numel(&shape)?;
        match node.op() {
            Op::Const
            | Op::Reshape
            | Op::Expand
            | Op::Permute
            | Op::Shrink
            | Op::Flip
            | Op::Load
            | Op::After => continue,
            Op::Window => continue,
            Op::Param => {
                let Arg::Param(p) = node.arg() else {
                    unreachable!()
                };
                e.types.insert(name.clone(), dt);
                e.decl += &format!(
                    "{} *{name}=({}*)inputs[{}];\n",
                    ctype(dt),
                    ctype(dt),
                    p.slot
                );
                continue;
            }
            _ => (),
        }
        e.alloc(&name, dt, n)?;
        if let Some(&(matmul, a, b, ref product)) = contractions.get(&v) {
            let rank = product.len();
            let (m, cols, k) = (product[rank - 3], product[rank - 2], product[rank - 1]);
            let batch = if rank == 4 { product[0] } else { 1 };
            if m == 0 || cols == 0 {
                e.material.insert(v);
                continue;
            }
            let ash = e.shape(a);
            let bsh = e.shape(b);
            let operand_coords = |shape: &[usize], left: bool| {
                let mut coords = if left {
                    vec!["row+ii".into(), "0".into(), "r".into()]
                } else {
                    vec!["0".into(), "col+jj".into(), "r".into()]
                };
                if rank == 4 {
                    coords.insert(
                        0,
                        if shape[0] == 1 {
                            "0".into()
                        } else {
                            "batch".into()
                        },
                    );
                }
                coords
            };
            let ar = e.read_coordinates(a, &operand_coords(&ash, true))?;
            let br = e.read_coordinates(b, &operand_coords(&bsh, false))?;
            e.overrides.insert(matmul, "acc[ii][jj]".into());
            let epilogue = e.read(v, "out_index")?;
            e.overrides.remove(&matmul);
            let expressions = format!("{ar} {br} {epilogue} {name}");
            let mut vars: Vec<_> = expressions
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .filter(|word| e.types.contains_key(*word))
                .map(str::to_owned)
                .collect();
            vars.sort();
            vars.dedup();
            let id = e.kernel_count;
            let fields = vars
                .iter()
                .map(|var| format!("{} *{var};", ctype(e.types[var])))
                .collect::<Vec<_>>()
                .join("\n");
            let locals = vars
                .iter()
                .map(|var| format!("{} *restrict {var}=ctx->{var};", ctype(e.types[var])))
                .collect::<Vec<_>>()
                .join("\n");
            let kernel = gemm::generate(
                id,
                m,
                cols,
                k,
                batch,
                &ar,
                &br,
                e.storage_strides(b)
                    .is_some_and(|s| matches!(s[rank - 2], 0 | 1)),
                &name,
                &format!("{name}[out_index]={epilogue}"),
                &fields,
                &locals,
                &vars.join(","),
                e.profile,
            );
            e.peak_packed_workspace = e.peak_packed_workspace.max(kernel.packed_bytes);
            e.kernels += &kernel.worker;
            let begin = e.body.len();
            e.body += &kernel.dispatch;
            e.kernel(
                v,
                begin,
                kernel.packed_bytes,
                2 * batch * m * cols * k,
                kernel.phase_timing_kind,
            );
            e.material.insert(v);
            continue;
        }
        let begin = e.body.len();
        let work = reductions.get(&v).copied().unwrap_or(v);
        let work_node = g.node(work)?;
        match work_node.op() {
            Op::Stack => {
                let part = src
                    .first()
                    .map(|&s| numel(&e.shape(s)))
                    .transpose()?
                    .unwrap_or(0);
                for (j, &s) in src.iter().enumerate() {
                    let x = e.read(s, "i")?;
                    e.body +=
                        &format!("for(size_t i=0;i<{part};i++) {name}[{}+i]={x};\n", j * part);
                }
            }
            Op::Reduce => {
                let Arg::Reduce { op, num_axes } = work_node.arg() else {
                    unreachable!()
                };
                let count = numel(&e.shape(work_node.src()[0])[..*num_axes])?;
                let x = e.read(work_node.src()[0], &format!("r*{n}+i"))?;
                let init = match op {
                    ReduceOp::Add => "0",
                    ReduceOp::Mul => "1",
                    ReduceOp::Max => match dt {
                        DType::F32 | DType::WeakFloat => "-INFINITY",
                        DType::I32 => "INT32_MIN",
                        DType::WeakInt => "INT64_MIN",
                        _ => "0",
                    },
                };
                let expr = match op {
                    ReduceOp::Add => format!("acc+({x})"),
                    ReduceOp::Mul => format!("acc*({x})"),
                    ReduceOp::Max => format!("acc>({x})?acc:({x})"),
                };
                e.overrides.insert(work, "acc".into());
                let epilogue = e.read(v, "i")?;
                e.overrides.remove(&work);
                e.body+=&format!("for(size_t i=0;i<{n};i++) {{ {} acc={init}; for(size_t r=0;r<{count};r++) acc={expr}; {name}[i]={epilogue}; }}\n",ctype(dt));
            }
            Op::Index => {
                let base = e.shape(src[0]);
                let row = numel(&base[1..])?;
                let ix = e.read(src[1], &format!("i/{}", row.max(1)))?;
                let val = e.read(src[0], &format!("ix*{row}+i%{}", row.max(1)))?;
                e.body+=&format!("for(size_t i=0;i<{n};i++) {{ int64_t ix={ix};if(ix<0 || ix>={}) {{err=2;goto cleanup;}} {name}[i]={val};}}\n",base[0]);
            }
            Op::Pad => {
                let old = e.shape(src[0]);
                let offsets = e.dims(src[1])?;
                let st = strides(&old);
                let check = old
                    .iter()
                    .enumerate()
                    .map(|(j, d)| {
                        let c = coord("i", &shape, j);
                        format!("{c}>={} && {c}<{}", offsets[j], offsets[j] + d)
                    })
                    .collect::<Vec<_>>()
                    .join(" && ");
                let idx = old
                    .iter()
                    .enumerate()
                    .map(|(j, _)| format!("({}-{})*{}", coord("i", &shape, j), offsets[j], st[j]))
                    .collect::<Vec<_>>()
                    .join("+");
                let x = e.read(src[0], &if idx.is_empty() { "0".into() } else { idx })?;
                e.body += &format!(
                    "for(size_t i=0;i<{n};i++) {name}[i]=({})?{x}:0;\n",
                    if check.is_empty() { "1" } else { &check }
                );
            }
            _ => {
                let expr = e.elementwise(v, "i")?;
                e.body += &format!("for(size_t i=0;i<{n};i++) {name}[i]={expr};\n");
            }
        }
        e.kernel(v, begin, 0, 0, "wall");
        e.material.insert(v);
    }
    let mut output_bytes = 0usize;
    if e.profile {
        e.body += "uint64_t output_start=pup_clock_ns();\n";
    }
    for (j, v) in outputs.iter().copied().enumerate() {
        let node = g.node(v)?;
        if !matches!(
            node.dtype(),
            DType::F32 | DType::I32 | DType::Bool | DType::U8
        ) {
            return Err(Error(
                "CPU outputs require concrete dtypes; add cast".into(),
            ));
        }
        let n = numel(&e.shape(v))?;
        let bytes = n
            .checked_mul(if matches!(node.dtype(), DType::Bool | DType::U8) {
                1
            } else {
                4
            })
            .ok_or_else(|| Error("CPU output size overflow".into()))?;
        output_bytes = output_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error("CPU workspace overflow".into()))?;
        if output_bytes > 2 * 1024 * 1024 * 1024 {
            return Err(Error(
                "CPU workspace exceeds 2 GiB including outputs".into(),
            ));
        }
        if e.direct_outputs.get(&e.names[&v]) == Some(&j) {
            continue;
        }
        let x = e.read(v, "i")?;
        e.mark_reads(&x, e.kernel_count);
        e.body += &format!(
            "for(size_t i=0;i<{n};i++) (({}*)outputs[{j}])[i]={x};\n",
            ctype(node.dtype())
        );
    }
    if e.profile {
        e.body += "stats[2]=pup_clock_ns()-output_start;\n";
    }
    let memory_plan = e.plan_memory()?;
    if e.peak_workspace
        .checked_add(e.peak_packed_workspace)
        .and_then(|bytes| bytes.checked_add(output_bytes))
        .is_none_or(|bytes| bytes > 2 * 1024 * 1024 * 1024)
    {
        return Err(Error(
            "CPU workspace exceeds 2 GiB including outputs".into(),
        ));
    }
    let max_tasks = fused
        .values()
        .map(|(_, _, product)| {
            let rank = product.len();
            product[rank - 3].div_ceil(4)
                * product[rank - 2].div_ceil(8)
                * if rank == 4 { product[0] } else { 1 }
        })
        .max()
        .unwrap_or(1)
        .max(e.parallel_tasks);
    if e.profile {
        let count = 4 + e.kernel_count * 4;
        let mut inputs = std::collections::BTreeMap::new();
        for &v in &order {
            if let Arg::Param(p) = g.node(v)?.arg() {
                inputs.insert(p.slot, serde_json::json!({"slot": p.slot, "dtype": p.dtype, "elements": p.size.unwrap_or(1)}));
            }
        }
        let output_metadata: Vec<_> = outputs.iter().map(|&v| {
            let node = g.node(v).unwrap();
            serde_json::json!({"dtype": node.dtype(), "shape": node.shape(), "bindings": e.labels.get(&v).cloned().unwrap_or_default()})
        }).collect();
        let metadata = serde_json::json!({
            "version": 1, "backend": "c_cpu", "reachable_pops": order.len(), "workspace_bytes": e.peak_workspace.max(1), "packed_workspace_bytes": e.peak_packed_workspace,
            "memory_plan": memory_plan, "matmul_stack_scratch_bytes_per_worker": if fused.is_empty() {0} else {320}, "stats_length": count, "header": ["invocation_ns", "setup_ns", "output_copy_ns", "cleanup_ns"],
            "kernel_fields": ["calls", "elapsed_ns", "packing_ns", "compute_ns"],
            "inputs": inputs.into_values().collect::<Vec<_>>(), "outputs": output_metadata, "kernels": e.kernel_metadata
        });
        let metadata_literal =
            serde_json::to_string(&metadata.to_string()).map_err(|e| Error(e.to_string()))?;
        let code = format!(
            r#"#define _POSIX_C_SOURCE 200809L
#include <time.h>
{runtime}
static uint64_t pup_clock_ns(void) {{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000000ULL+(uint64_t)ts.tv_nsec;
}}
{kernels}
const char *pup_profile_metadata(void) {{ return {metadata_literal}; }}
int pup_run_profiled(const void **inputs,void **outputs,size_t threads,uint64_t *stats,size_t stats_length) {{
    if (!stats || stats_length < {count}) return 4;
    for (size_t i=0;i<{count};i++) stats[i]=0;
    uint64_t invocation_start=pup_clock_ns();
    if (!threads) {{ stats[0]=pup_clock_ns()-invocation_start; return 3; }}
    int err=0;
    {decl}
    unsigned char *arena=malloc({workspace}ULL);
    if (!arena) {{ stats[0]=pup_clock_ns()-invocation_start; return 1; }}
    pup_pool pool;
    if ((err=pup_pool_init(&pool,threads<{max_tasks}ULL?threads:{max_tasks}ULL))) {{
        free(arena); stats[0]=pup_clock_ns()-invocation_start; return err;
    }}
    stats[1]=pup_clock_ns()-invocation_start;
    {body}
cleanup:;
    uint64_t cleanup_start=pup_clock_ns();
    pup_pool_destroy(&pool);
    free(arena);
    stats[3]=pup_clock_ns()-cleanup_start;
    stats[0]=pup_clock_ns()-invocation_start;
    return err;
}}
int pup_run(const void **inputs,void **outputs,size_t threads) {{
    uint64_t stats[{count}];
    return pup_run_profiled(inputs,outputs,threads,stats,{count});
}}
"#,
            runtime = include_str!("cpu_runtime.c"),
            kernels = e.kernels,
            decl = e.decl,
            workspace = e.peak_workspace.max(1),
            body = e.body
        );
        return Ok((code, fused.len()));
    }
    Ok((format!("// puppygrad C ABI 3: self-contained C, signed arithmetic wraps (-fwrapv)\n{}\n{}\nint pup_run(const void **inputs,void **outputs,size_t threads) {{\nif(!threads) return 3;\nint err=0;\n{}unsigned char *arena=malloc({}ULL); if(!arena) return 1;\npup_pool pool;\nif((err=pup_pool_init(&pool,threads<{max_tasks}ULL?threads:{max_tasks}ULL))) {{free(arena);return err;}}\n{}cleanup:\npup_pool_destroy(&pool);\nfree(arena);return err;\n}}\n",include_str!("cpu_runtime.c"),e.kernels,e.decl,e.peak_workspace.max(1),e.body),fused.len()))
}

pub fn compile(g: &Graph, root: Value, cache: &Path) -> Result<Executable> {
    compile_with_options(g, root, cache, &BuildOptions::default())
}

pub fn compile_with_options(
    g: &Graph,
    root: Value,
    cache: &Path,
    options: &BuildOptions,
) -> Result<Executable> {
    let (source, gemm_count) = emit(g, root)?;
    compile_source(g, root, cache, source, gemm_count, false, options)
}

pub fn compile_profiled(program: &super::source::Program, cache: &Path) -> Result<Executable> {
    compile_profiled_with_options(program, cache, &BuildOptions::default())
}

pub fn compile_profiled_with_options(
    program: &super::source::Program,
    cache: &Path,
    options: &BuildOptions,
) -> Result<Executable> {
    let (source, gemm_count) = emit_profiled(program)?;
    compile_source(
        &program.graph,
        program.root,
        cache,
        source,
        gemm_count,
        true,
        options,
    )
}

fn compile_source(
    g: &Graph,
    root: Value,
    cache: &Path,
    source: String,
    gemm_count: usize,
    profile: bool,
    options: &BuildOptions,
) -> Result<Executable> {
    use std::hash::{Hash, Hasher};
    let build_info = BuildInfo::resolve(options)?;
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hash);
    build_info.hash(&mut hash);
    let key = format!("{:016x}", hash.finish());
    std::fs::create_dir_all(cache).map_err(|e| Error(e.to_string()))?;
    let cache = cache.canonicalize().map_err(|e| Error(e.to_string()))?;
    let source_path = cache.join(format!("{key}.c"));
    let library_path = cache.join(format!("{key}.so"));
    let manifest_path = cache.join(format!("{key}.json"));
    let valid = std::fs::read_to_string(&source_path).is_ok_and(|s| s == source)
        && library_path.is_file()
        && std::fs::read(&manifest_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<BuildInfo>(&b).ok())
            .as_ref()
            == Some(&build_info);
    if !valid {
        // Each compiler invocation gets private temporary files, then atomic publication.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = format!(
            "{key}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let c = cache.join(format!("{unique}.c"));
        let so = cache.join(format!("{unique}.so"));
        let manifest = cache.join(format!("{unique}.json"));
        std::fs::write(&c, &source).map_err(|e| Error(e.to_string()))?;
        let result = std::process::Command::new(&build_info.compiler_path)
            .args(&build_info.flags)
            .arg("-o")
            .arg(&so)
            .arg(&c)
            .arg("-lm")
            .output()
            .map_err(|e| Error(format!("C CPU backend needs cc: {e}")))?;
        if !result.status.success() {
            return Err(Error(format!(
                "C compilation failed: {}",
                String::from_utf8_lossy(&result.stderr)
            )));
        }
        std::fs::write(
            &manifest,
            serde_json::to_vec_pretty(&build_info).map_err(|e| Error(e.to_string()))?,
        )
        .map_err(|e| Error(e.to_string()))?;
        std::fs::rename(&so, &library_path).map_err(|e| Error(e.to_string()))?;
        std::fs::rename(&c, &source_path).map_err(|e| Error(e.to_string()))?;
        std::fs::rename(&manifest, &manifest_path).map_err(|e| Error(e.to_string()))?;
    }
    let library =
        unsafe { libloading::Library::new(&library_path) }.map_err(|e| Error(e.to_string()))?;
    let run = *unsafe { library.get::<Run>(b"pup_run\0") }.map_err(|e| Error(e.to_string()))?;
    let (profile_run, profile_metadata) = if profile {
        let run = *unsafe { library.get::<ProfileRun>(b"pup_run_profiled\0") }
            .map_err(|e| Error(e.to_string()))?;
        let metadata = unsafe {
            library
                .get::<unsafe extern "C" fn() -> *const std::ffi::c_char>(b"pup_profile_metadata\0")
        }
        .map_err(|e| Error(e.to_string()))?;
        let bytes = unsafe { std::ffi::CStr::from_ptr(metadata()) }.to_bytes();
        let mut metadata: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| Error(e.to_string()))?;
        metadata["build"] = serde_json::to_value(&build_info).map_err(|e| Error(e.to_string()))?;
        (Some(run), Some(metadata))
    } else {
        (None, None)
    };
    let inputs = g
        .toposort(root)?
        .iter()
        .filter_map(|&v| match g.node(v).unwrap().arg() {
            Arg::Param(p) => Some((p.slot, p.dtype, p.size.unwrap_or(1))),
            _ => None,
        })
        .collect();
    let outs = if g.node(root)?.op() == Op::Sink {
        g.node(root)?.src().to_vec()
    } else {
        vec![root]
    };
    let outputs = outs
        .iter()
        .map(|&v| {
            let n = g.node(v).unwrap();
            (n.dtype(), n.shape().unwrap().to_vec())
        })
        .collect();
    Ok(Executable {
        _library: library,
        run,
        profile_run,
        profile_metadata,
        cache_hit: valid,
        build_info,
        inputs,
        state_slots: g
            .toposort(root)?
            .iter()
            .filter_map(|&v| match g.node(v).unwrap().arg() {
                Arg::Param(p) if p.writable => Some(p.slot),
                _ => None,
            })
            .collect(),
        runtime: Runtime::default(),
        outputs,
        gemm_count,
        source_path,
    })
}
