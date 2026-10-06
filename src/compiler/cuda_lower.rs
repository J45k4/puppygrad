//! CUDA scheduling reuses the C backend's view/expression semantics and arena planner.
//! Contractions and row reductions use shape-selected parallel schedules.
use super::*;
use crate::compiler::gpu::{Binding, Kernel, Lowered};
#[path = "cuda_fuse.rs"]
mod expression_fusion;
#[path = "cuda_reduce.rs"]
mod reduction;

pub(crate) fn emit_backend(
    g: &Graph,
    root: Value,
    backend: crate::compiler::gpu::Backend,
) -> Result<Lowered> {
    let outputs = if g.node(root)?.op() == Op::Sink {
        g.node(root)?.src().to_vec()
    } else {
        vec![root]
    };
    let order = g.toposort(root)?;
    let mut uses = HashMap::<Value, usize>::new();
    for &v in &order {
        for &s in g.node(v)?.src() {
            *uses.entry(s).or_default() += 1;
        }
    }
    for &out in &outputs {
        *uses.entry(out).or_default() += 1;
    }
    let mut fused = HashMap::new();
    let mut skip = HashSet::new();
    let mut store_values = HashSet::new();
    for &v in &order {
        if g.node(v)?.op() == Op::Store {
            let target = g.node(v)?.src()[0];
            store_values.insert(g.node(v)?.src()[1]);
            if !outputs.contains(&target)
                && order.iter().all(|&consumer| {
                    let node = g.node(consumer).unwrap();
                    !node.src().contains(&target)
                        || (node.op() == Op::Store
                            && node.src()[0] == target
                            && node.src()[1] != target)
                })
            {
                skip.insert(target);
            }
        }
    }
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
    let parallel_reductions = std::env::var(backend.env("REDUCTIONS")).as_deref() != Ok("0");
    let register_tiles = std::env::var(backend.env("REGISTER_TILES")).as_deref() != Ok("0");
    let row_fusions =
        if parallel_reductions && std::env::var(backend.env("ROW_FUSION")).as_deref() != Ok("0") {
            reduction::plan(g, &order, &outputs, &consumers, &skip)
        } else {
            HashMap::new()
        };
    for fusion in row_fusions.values() {
        skip.extend(fusion.inner.iter().copied());
    }
    let fuse_expressions = std::env::var(backend.env("EXPRESSIONS")).as_deref() != Ok("0");
    let epilogues = if fuse_expressions {
        expression_fusion::epilogues(g, &order, &outputs, &consumers, &mut fused, &mut skip)?
    } else {
        HashMap::new()
    };
    // Shape operands are compile-time metadata, not executable tensor work.
    for &v in &order {
        if g.node(v)?.op() == Op::Stack
            && !outputs.contains(&v)
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
    // Inline bounded, single-use pointwise expressions into consumers through
    // views. Shared values stay materialized to avoid repeated residual chains.
    let mut contraction_inputs = HashSet::new();
    for &(a, b, _) in fused.values() {
        for mut v in [a, b] {
            loop {
                let node = g.node(v)?;
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
                    v = node.src()[0];
                } else {
                    break;
                }
            }
            contraction_inputs.insert(v);
        }
    }
    let mut costs = HashMap::<Value, usize>::new();
    let mut inline_values = HashSet::new();
    let mut constant_values = HashSet::new();
    let mut ranges = HashMap::new();
    for &v in &order {
        let node = g.node(v)?;
        let constant = node.op() == Op::Const
            || (pointwise(node.op())
                && node.shape().is_some_and(|s| s.is_empty())
                && node.src().iter().all(|s| constant_values.contains(s)));
        if constant {
            constant_values.insert(v);
        }
        if fuse_expressions && !outputs.contains(&v) && !skip.contains(&v) {
            if let Some(range) = expression_fusion::integer_range(g, v) {
                ranges.insert(v, range);
                inline_values.insert(v);
                continue;
            }
        }
        let cost = 1 + node
            .src()
            .iter()
            .map(|s| costs.get(s).copied().unwrap_or(0))
            .sum::<usize>()
            .min(64);
        if pointwise(node.op())
            && cost <= 24
            && (uses.get(&v).copied().unwrap_or(0) <= 1 || (fuse_expressions && constant))
            && !outputs.contains(&v)
            && !store_values.contains(&v)
            && !skip.contains(&v)
            && !contraction_inputs.contains(&v)
            && !row_fusions.contains_key(&v)
            && !epilogues.contains_key(&v)
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
    if fuse_expressions {
        expression_fusion::store_stacks(
            g,
            &order,
            &outputs,
            &consumers,
            &skip,
            &mut inline_values,
        )?;
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
        profile: false,
        wrap_signed_arithmetic: true,
        labels: HashMap::new(),
        kernel_metadata: Vec::new(),
    };
    for (v, (first, step)) in ranges {
        e.overrides.insert(
            v,
            format!(
                "(({})(INT64_C({first}) + ((int64_t)($index))*INT64_C({step})))",
                ctype(g.node(v)?.dtype())
            ),
        );
    }

    // A materialized graph result can write directly to the caller output.
    // Duplicate outputs still get a separate copy; views/constants/parameters
    // retain their explicit final-output kernel.
    for (slot, &v) in outputs.iter().enumerate() {
        let op = g.node(v)?.op();
        if !matches!(
            op,
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
            e.direct_outputs.entry(e.names[&v].clone()).or_insert(slot);
        }
    }
    let mut kernels = Vec::new();
    let mut parallel_reduction_count = 0;
    let mut inputs = std::collections::BTreeMap::new();
    let mut output_specs = Vec::new();
    for &v in &outputs {
        let node = g.node(v)?;
        if !matches!(
            node.dtype(),
            DType::F32 | DType::I32 | DType::Bool | DType::U8
        ) {
            return Err(Error(
                "CUDA outputs require concrete dtypes; add cast".into(),
            ));
        }
        output_specs.push((node.dtype(), e.shape(v)));
    }
    for &v in &order {
        let node = g.node(v)?;
        let name = e.names[&v].clone();
        let src = node.src();
        let dt = node.dtype();
        if skip.contains(&v) || e.inline_values.contains(&v) || node.op() == Op::Sink {
            continue;
        }
        if let Arg::Param(p) = node.arg() {
            inputs.insert(p.slot, (p.slot, dt, p.size.unwrap_or(1)));
            e.types.insert(name, dt);
            continue;
        }
        if matches!(
            node.op(),
            Op::Const
                | Op::Reshape
                | Op::Expand
                | Op::Permute
                | Op::Shrink
                | Op::Flip
                | Op::Load
                | Op::After
                | Op::Window
        ) {
            continue;
        }
        if node.op() == Op::Store {
            let payload = src[1];
            let n = numel(&e.shape(payload))?;
            let value = if e.store_snapshot_needed(payload) {
                let snapshot = format!("state_snapshot{}", kernels.len());
                e.alloc(&snapshot, g.node(payload)?.dtype(), n)?;
                let x = e.read(payload, "i")?;
                add_kernel(&mut e, &mut kernels, n, format!("{snapshot}[i]={x};"));
                format!("{snapshot}[i]")
            } else {
                e.read(payload, "i")?
            };
            let (_, body) = e.store_parts(v, &value, true)?;
            let destination = g.node(src[0])?;
            if g.unique_indices(destination.src()[1]) {
                add_kernel(&mut e, &mut kernels, n, body);
            } else {
                add_kernel_with_grid(
                    &mut e,
                    &mut kernels,
                    n,
                    format!("if(threadIdx.x)return;for(size_t i=0;i<{n};i++){{{body}}}"),
                    Some(1),
                );
            }
            continue;
        }
        let shape = e.shape(v);
        let n = numel(&shape)?;
        e.alloc(&name, dt, n)?;
        let mut tiled_blocks = None;
        let code = if let Some(fusion) = row_fusions.get(&v) {
            tiled_blocks = Some(fusion.rows.div_ceil(8));
            fusion.code(&mut e, v, &name)?
        } else if let Some(&(a, b, ref product)) = fused.get(&v) {
            let rank = product.len();
            let cols = product[rank - 2];
            let m = product[rank - 3];
            let k = product[rank - 1];
            let original = epilogues.get(&v).copied().unwrap_or(v);
            e.overrides.insert(original, "acc".into());
            let epilogue = e.read(v, "i")?;
            let tiled_epilogue = e.read(v, &format!("((batch*{m}+out_row)*{cols}+out_col)"))?;
            e.overrides.remove(&original);
            let coords = |value, left| {
                let sh = e.shape(value);
                let mut cs = if left {
                    vec!["row".into(), "0".into(), "r".into()]
                } else {
                    vec!["0".into(), "col".into(), "r".into()]
                };
                if rank == 4 {
                    cs.insert(
                        0,
                        if sh[0] == 1 {
                            "0".into()
                        } else {
                            "batch".into()
                        },
                    );
                }
                cs
            };
            let ar = e.read_coordinates(a, &coords(a, true))?;
            let br = e.read_coordinates(b, &coords(b, false))?;
            if register_tiles && m >= 32 && cols >= 64 && k >= 32 {
                // Eight outputs per thread reuse each shared operand. Pad the
                // shared rows so transposed loads and row broadcasts do not
                // collide in the same memory banks on CUDA or HIP hardware.
                let tiles_m = m.div_ceil(32);
                let tiles_n = cols.div_ceil(64);
                tiled_blocks = Some(n / (m * cols) * tiles_m * tiles_n);
                let contiguous_k = e.storage_strides(b).is_some_and(|st| st[rank - 1] == 1);
                let b_coords = if contiguous_k {
                    "size_t col=tile_col+q/32,r=base+q%32;"
                } else {
                    "size_t col=tile_col+q%64,r=base+q/64;"
                };
                let b_store = if contiguous_k {
                    "(q%32)*65+q/32"
                } else {
                    "(q/64)*65+q%64"
                };
                format!(
                    "// register-tiled contraction: 32 rows, 64 columns\n\
__shared__ float sa[32*33],sb[32*65];
const size_t tid=threadIdx.x,tile_col=(blockIdx.x%{tiles_n})*64,
 tile_row=((blockIdx.x/{tiles_n})%{tiles_m})*32,batch=blockIdx.x/{};
float sums[2][4]={{}};
for(size_t base=0;base<{k};base+=32) {{
 for(size_t q=tid;q<1024;q+=256) {{
  size_t row=tile_row+q/32,r=base+q%32;
  sa[(q/32)*33+q%32]=(row<{m} && r<{k})?({ar}):0.0f;
 }}
 for(size_t q=tid;q<2048;q+=256) {{
  {b_coords} sb[{b_store}]=(col<{cols} && r<{k})?({br}):0.0f;
 }}
 __syncthreads();
 #pragma unroll
 for(size_t r=0;r<32;r++) {{
  if(base+r<{k}) {{
   #pragma unroll
   for(size_t mr=0;mr<2;mr++) {{
    float av=sa[(tid/16+mr*16)*33+r];
    #pragma unroll
    for(size_t nc=0;nc<4;nc++) sums[mr][nc]+=av*sb[r*65+tid%16+nc*16];
   }}
  }}
 }}
 __syncthreads();
}}
#pragma unroll
for(size_t mr=0;mr<2;mr++) {{
 const size_t out_row=tile_row+tid/16+mr*16;
 #pragma unroll
 for(size_t nc=0;nc<4;nc++) {{
  const size_t out_col=tile_col+tid%16+nc*16;
  if(out_row<{m} && out_col<{cols}) {{
   const float acc=sums[mr][nc];
   {name}[(batch*{m}+out_row)*{cols}+out_col]={tiled_epilogue};
  }}
 }}
}}",
                    tiles_m * tiles_n
                )
            } else if m >= 16 && cols >= 16 && k >= 32 {
                let tiles_m = m.div_ceil(16);
                let tiles_n = cols.div_ceil(16);
                tiled_blocks = Some(n / (m * cols) * tiles_m * tiles_n);
                // Use the physical view layout to coalesce either B's columns
                // or reduction axis; window/padded/non-affine views remain valid.
                let contiguous_k = e.storage_strides(b).is_some_and(|st| st[rank - 1] == 1);
                let b_coords = if contiguous_k {
                    "size_t col=tile_col+q/32,r=base+q%32;"
                } else {
                    "size_t col=tile_col+q%16,r=base+q/16;"
                };
                let b_store = if contiguous_k { "(q%32)*16+q/32" } else { "q" };
                format!(
                    "__shared__ float sa[16*32],sb[32*16];
const size_t tid=threadIdx.x,tile_col=(blockIdx.x%{tiles_n})*16,
 tile_row=((blockIdx.x/{tiles_n})%{tiles_m})*16,batch=blockIdx.x/{};
const size_t out_row=tile_row+tid/16,out_col=tile_col+tid%16;
float acc=0;
for(size_t base=0;base<{k};base+=32) {{
 for(size_t q=tid;q<512;q+=256) {{
  {{size_t row=tile_row+q/32,r=base+q%32;sa[q]=(row<{m} && r<{k})?({ar}):0.0f;}}
  {{{b_coords} sb[{b_store}]=(col<{cols} && r<{k})?({br}):0.0f;}}
 }}
 __syncthreads();
 #pragma unroll
 for(size_t r=0;r<32;r++) {{if(base+r<{k}) acc+=sa[(tid/16)*32+r]*sb[r*16+tid%16];}}
 __syncthreads();
}}
if(out_row<{m} && out_col<{cols}) {name}[(batch*{m}+out_row)*{cols}+out_col]={tiled_epilogue};",
                    tiles_m * tiles_n
                )
            } else if m > 0
                && cols > 0
                && k >= 32
                && e.storage_strides(b).is_some_and(|st| st[rank - 1] == 1)
            {
                // One warp per output: lanes read consecutive reduction elements.
                // This handles small matrices and vector products using the same
                // contraction recognition as the tiled matrix path.
                tiled_blocks = Some(n.div_ceil(8));
                format!("const size_t lane=threadIdx.x%32,i=(size_t)blockIdx.x*8+threadIdx.x/32; if(i>={n})return; const size_t col=i%{cols},row=(i/{cols})%{m},batch=i/{}; float acc=0; for(size_t r=lane;r<{k};r+=32) acc+=({ar})*({br}); for(int offset=16;offset>0;offset/=2) acc+=__shfl_down_sync(0xffffffff,acc,offset); if(lane==0) {name}[i]={epilogue};",m*cols)
            } else {
                format!("const size_t col=i%{},row=(i/{})%{},batch=i/{}; float acc=0; for(size_t r=0;r<{k};r++) acc+=({ar})*({br}); {name}[i]={epilogue};", cols.max(1), cols.max(1), m.max(1), (m*cols).max(1))
            }
        } else {
            match node.op() {
                Op::Stack => {
                    let part = src
                        .first()
                        .map(|&s| numel(&e.shape(s)))
                        .transpose()?
                        .unwrap_or(0);
                    let mut code = String::new();
                    for (j, &s) in src.iter().enumerate() {
                        let x = e.read(s, &format!("i-{}", j * part))?;
                        code += &format!(
                            "if(i>={} && i<{}) {name}[i]={x};\n",
                            j * part,
                            (j + 1) * part
                        );
                    }
                    code
                }
                Op::Reduce => {
                    let Arg::Reduce { op, num_axes } = node.arg() else {
                        unreachable!()
                    };
                    let count = numel(&e.shape(src[0])[..*num_axes])?;
                    let x = e.read(src[0], &format!("r*{n}+i"))?;
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
                        ReduceOp::Add if matches!(dt, DType::I32 | DType::WeakInt) => {
                            let unsigned = if dt == DType::I32 {
                                "unsigned int"
                            } else {
                                "unsigned long long"
                            };
                            format!("({})( ({unsigned})acc+({unsigned})({x}))", ctype(dt))
                        }
                        ReduceOp::Mul if matches!(dt, DType::I32 | DType::WeakInt) => {
                            let unsigned = if dt == DType::I32 {
                                "unsigned int"
                            } else {
                                "unsigned long long"
                            };
                            format!("({})( ({unsigned})acc*({unsigned})({x}))", ctype(dt))
                        }
                        ReduceOp::Add => format!("acc+({x})"),
                        ReduceOp::Mul => format!("acc*({x})"),
                        ReduceOp::Max => format!("acc>({x})?acc:({x})"),
                    };
                    let contiguous = *num_axes == 1
                        && e.storage_strides(src[0])
                            .is_some_and(|s| s.first() == Some(&1));
                    if parallel_reductions
                        && dt == DType::F32
                        && count >= 32
                        && (n < 256 || contiguous)
                    {
                        parallel_reduction_count += 1;
                        tiled_blocks = Some(n.div_ceil(8));
                        let reduce = reduction::warp_reduce(count, *op, &x, "acc");
                        format!("const size_t lane=threadIdx.x%32,i=(size_t)blockIdx.x*8+threadIdx.x/32; if(i>={n})return; {reduce} if(lane==0){name}[i]=acc;")
                    } else {
                        format!(
                        "{} acc={init}; for(size_t r=0;r<{count};r++) acc={expr}; {name}[i]=acc;",
                        ctype(dt)
                    )
                    }
                }
                Op::Index => {
                    let base = e.shape(src[0]);
                    let row = numel(&base[1..])?;
                    let ix = e.read(src[1], &format!("i/{}", row.max(1)))?;
                    let x = e.read(src[0], &format!("ix*{row}+i%{}", row.max(1)))?;
                    format!("int64_t ix={ix}; if(ix<0 || ix>={}) {{atomicExch(error,2);return;}} {name}[i]={x};",base[0])
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
                        .map(|(j, _)| {
                            format!("({}-{})*{}", coord("i", &shape, j), offsets[j], st[j])
                        })
                        .collect::<Vec<_>>()
                        .join("+");
                    let x = e.read(src[0], if idx.is_empty() { "0" } else { &idx })?;
                    format!(
                        "{name}[i]=({})?{x}:0;",
                        if check.is_empty() { "1" } else { &check }
                    )
                }
                _ => {
                    let x = e.elementwise(v, "i")?;
                    format!("{name}[i]={x};")
                }
            }
        };
        add_kernel_with_grid(&mut e, &mut kernels, n, code, tiled_blocks);
        e.material.insert(v);
    }
    for (slot, &v) in outputs.iter().enumerate() {
        if e.direct_outputs.get(&e.names[&v]) == Some(&slot) {
            continue;
        }
        let n = numel(&e.shape(v))?;
        let x = e.read(v, "i")?;
        let name = format!("out{slot}");
        e.types.insert(name.clone(), g.node(v)?.dtype());
        add_kernel(&mut e, &mut kernels, n, format!("{name}[i]={x};"));
    }
    e.plan_memory()?;
    let mut bindings = HashMap::new();
    for &v in &order {
        if let Arg::Param(p) = g.node(v)?.arg() {
            bindings.insert(e.names[&v].clone(), Binding::Input(p.slot));
        }
    }
    for a in &e.allocations {
        bindings.insert(a.name.clone(), Binding::Arena(a.offset));
    }
    for (name, &slot) in &e.direct_outputs {
        bindings.insert(name.clone(), Binding::Output(slot));
    }
    for slot in 0..outputs.len() {
        bindings.insert(format!("out{slot}"), Binding::Output(slot));
    }
    let mut source=String::from("// Puppygrad CUDA kernels: caller buffers, no framework math.\ntypedef long long int64_t;\ntypedef int int32_t;\ntypedef unsigned char uint8_t;\n#define INT64_C(x) x##LL\n#define INT32_MIN (-2147483647-1)\n#define INT64_MIN (-9223372036854775807LL-1)\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\n");
    if backend == crate::compiler::gpu::Backend::Hip {
        // HIPRTC supplies HIP device types and intrinsics. Avoid redeclaring
        // fixed-width types: HIPRTC headers keep those in an internal namespace.
        source = String::from("// Puppygrad HIP kernels: caller buffers, no framework math.\n#ifndef __HIPCC_RTC__\n#include <hip/hip_runtime.h>\n#endif\ntypedef long long pup_int64_t;\ntypedef int pup_int32_t;\ntypedef unsigned char pup_uint8_t;\n#ifndef INFINITY\n#define INFINITY (__int_as_float(0x7f800000))\n#endif\n#ifndef NAN\n#define NAN (__int_as_float(0x7fffffff))\n#endif\n#ifndef INT64_C\n#define INT64_C(x) x##LL\n#endif\n#ifndef INT32_MIN\n#define INT32_MIN (-2147483647-1)\n#endif\n#ifndef INT64_MIN\n#define INT64_MIN (-9223372036854775807LL-1)\n#endif\n");
    }
    if backend == crate::compiler::gpu::Backend::Hip {
        source += "template<typename T> __device__ __forceinline__ T pup_shfl_down(T x, unsigned int delta) { return __shfl_down(x, delta, 32); }\ntemplate<typename T> __device__ __forceinline__ T pup_shfl(T x, int lane) { return __shfl(x, lane, 32); }\n";
    }
    for k in &mut kernels {
        if backend == crate::compiler::gpu::Backend::Hip {
            // Explicit width=32 isolates logical subwarps on wave64 as well
            // as wave32 hardware; HIP's default width is device-dependent.
            k.code = k
                .code
                .replace("int64_t", "pup_int64_t")
                .replace("int32_t", "pup_int32_t")
                .replace("uint8_t", "pup_uint8_t")
                .replace("__shfl_down_sync(0xffffffff,", "pup_shfl_down(")
                .replace("__shfl_sync(0xffffffff,", "pup_shfl(");
        }
        for arg in &k.names {
            k.bindings.push(
                bindings
                    .get(arg)
                    .ok_or_else(|| Error(format!("missing CUDA storage: {arg}")))?
                    .clone(),
            );
        }
        source += &k.code;
    }
    let total_outputs = output_specs.iter().try_fold(0usize, |sum, (dt, shape)| {
        sum.checked_add(
            numel(shape)?
                .checked_mul(crate::compiler::gpu::dtype_bytes(*dt))
                .ok_or_else(|| Error("CUDA output overflow".into()))?,
        )
        .ok_or_else(|| Error("CUDA output overflow".into()))
    })?;
    if e.peak_workspace
        .checked_add(total_outputs)
        .is_none_or(|n| n > 2 * 1024 * 1024 * 1024)
    {
        return Err(Error(
            "CUDA workspace exceeds 2 GiB including outputs".into(),
        ));
    }
    Ok(Lowered {
        source,
        kernels,
        inputs: inputs.into_values().collect(),
        state_slots: order
            .iter()
            .filter_map(|&v| match g.node(v).unwrap().arg() {
                Arg::Param(p) if p.writable => Some(p.slot),
                _ => None,
            })
            .collect(),
        outputs: output_specs,
        workspace_bytes: e.peak_workspace.max(1),
        gemm_count: fused.len(),
        row_fusion_count: row_fusions.len(),
        parallel_reduction_count,
    })
}

fn add_kernel(e: &mut Emitter<'_>, kernels: &mut Vec<Kernel>, n: usize, code: String) {
    add_kernel_with_grid(e, kernels, n, code, None);
}
fn add_kernel_with_grid(
    e: &mut Emitter<'_>,
    kernels: &mut Vec<Kernel>,
    n: usize,
    code: String,
    tiled_blocks: Option<usize>,
) {
    let mut names = code
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|x| e.types.contains_key(*x))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    let params = names
        .iter()
        .map(|v| format!("{} *{v}", ctype(e.types[v])))
        .chain(std::iter::once("int *error".into()))
        .collect::<Vec<_>>()
        .join(", ");
    let id = kernels.len();
    e.mark_reads(&code, id);
    e.kernel_count = id + 1;
    // Only gather kernels can write the flag during this launch. Their reads
    // remain atomic; all other kernels see a stable value from the prior kernel.
    let error_guard = if code.contains("atomicExch(error") {
        "if(atomicAdd(error,0)) return;"
    } else {
        "if(*error) return;"
    };
    let index = if tiled_blocks.is_some() {
        String::new()
    } else {
        format!("size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>={n}ULL)return;\n")
    };
    kernels.push(Kernel {
        names, bindings: vec![], elements: n, blocks: tiled_blocks.unwrap_or(n.div_ceil(256)),
        code: format!("extern \"C\" __global__ void kernel{id}({params}) {{\n{error_guard}\n{index}{code}\n}}\n"),
    });
}
