//! Row-local reduction schedules and fusion of primitive normalization graphs.
use super::*;

pub(super) struct Fusion {
    pub rows: usize,
    pub width: usize,
    pub inner: HashSet<Value>,
    kind: Kind,
}
enum Kind {
    Rms { sum: Value },
    Softmax { max: Value, sum: Value, exps: Value },
}
fn args(g: &Graph, v: Value, op: Op) -> Option<&[Value]> {
    let n = g.node(v).ok()?;
    (n.op() == op).then_some(n.src())
}
fn row_reduce(
    g: &Graph,
    v: Value,
    op: ReduceOp,
    rows: usize,
    width: usize,
) -> Option<(Value, Value)> {
    let n = g.node(v).ok()?;
    if n.dtype() != DType::F32
        || n.arg() != &(Arg::Reduce { op, num_axes: 1 })
        || n.shape()? != [rows]
    {
        return None;
    }
    let p = n.src()[0];
    let pn = g.node(p).ok()?;
    if pn.op() != Op::Permute || pn.arg() != &Arg::Axes(vec![1, 0]) {
        return None;
    }
    let x = pn.src()[0];
    (g.node(x).ok()?.shape()? == [rows, width]).then_some((x, p))
}
fn row_broadcast(g: &Graph, v: Value, rows: usize) -> Option<Value> {
    let n = g.node(v).ok()?;
    if n.op() != Op::Reshape
        || n.shape()? != [rows, 1]
        || g.node(n.src()[0]).ok()?.shape()? != [rows]
    {
        return None;
    }
    Some(n.src()[0])
}
fn recognize(g: &Graph, root: Value) -> Option<Fusion> {
    let node = g.node(root).ok()?;
    let sh = node.shape()?;
    if node.dtype() != DType::F32 || sh.len() != 2 || sh[1] < 32 || sh[1] > 65536 {
        return None;
    }
    let (rows, width) = (sh[0], sh[1]);
    // exps / broadcast(sum(exps)), with exps = exp2((x-broadcast(max(x)))*scale).
    if let Some(div) = args(g, root, Op::Fdiv) {
        let exps = div[0];
        if let Some(sum) = row_broadcast(g, div[1], rows) {
            if let Some((input, sp)) = row_reduce(g, sum, ReduceOp::Add, rows, width) {
                if input == exps {
                    let scaled = args(g, exps, Op::Exp2)?[0];
                    let mul = args(g, scaled, Op::Mul)?;
                    // The scale must be a scalar, so the max-subtraction is row-local.
                    for (centered, scale) in [(mul[0], mul[1]), (mul[1], mul[0])] {
                        if numel(g.node(scale).ok()?.shape()?).ok()? != 1 {
                            continue;
                        }
                        let Some(sub) = args(g, centered, Op::Sub) else {
                            continue;
                        };
                        let Some(max) = row_broadcast(g, sub[1], rows) else {
                            continue;
                        };
                        let Some((x, mp)) = row_reduce(g, max, ReduceOp::Max, rows, width) else {
                            continue;
                        };
                        if x != sub[0] {
                            continue;
                        }
                        return Some(Fusion {
                            rows,
                            width,
                            inner: [div[1], sum, sp, exps, scaled, centered, sub[1], max, mp]
                                .into_iter()
                                .collect(),
                            kind: Kind::Softmax { max, sum, exps },
                        });
                    }
                }
            }
        }
    }
    // Optional affine weight after x / sqrt(broadcast(sum(x*x)/scale)+epsilon).
    let candidates = if let Some(mul) = args(g, root, Op::Mul) {
        vec![mul[0], mul[1]]
    } else {
        vec![root]
    };
    for div in candidates {
        let Some(ds) = args(g, div, Op::Fdiv) else {
            continue;
        };
        let x = ds[0];
        if g.node(x).ok()?.shape()? != [rows, width] {
            continue;
        }
        let Some(sq) = args(g, ds[1], Op::Sqrt) else {
            continue;
        };
        let Some(add) = args(g, sq[0], Op::Add) else {
            continue;
        };
        for (mean, epsilon) in [(add[0], add[1]), (add[1], add[0])] {
            if numel(g.node(epsilon).ok()?.shape()?).ok()? != 1 {
                continue;
            }
            let Some(average) = row_broadcast(g, mean, rows) else {
                continue;
            };
            let Some(av) = args(g, average, Op::Fdiv) else {
                continue;
            };
            if numel(g.node(av[1]).ok()?.shape()?).ok()? != 1 {
                continue;
            }
            let sum = av[0];
            let Some((square, p)) = row_reduce(g, sum, ReduceOp::Add, rows, width) else {
                continue;
            };
            if args(g, square, Op::Mul) != Some(&[x, x][..]) {
                continue;
            }
            let mut inner: HashSet<_> = [div, ds[1], sq[0], mean, average, sum, p, square]
                .into_iter()
                .collect();
            inner.remove(&root);
            return Some(Fusion {
                rows,
                width,
                inner,
                kind: Kind::Rms { sum },
            });
        }
    }
    None
}
pub(super) fn plan(
    g: &Graph,
    order: &[Value],
    outputs: &[Value],
    consumers: &HashMap<Value, Vec<Value>>,
    skip: &HashSet<Value>,
) -> HashMap<Value, Fusion> {
    let positions: HashMap<_, _> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut plans = HashMap::new();
    let mut reserved = HashSet::new();
    for &v in order.iter().rev() {
        let Some(p) = recognize(g, v) else { continue };
        // Fusion delays its intermediate reads until the root. Do not move
        // those reads across writable-state effects, even for an unrelated slot.
        let first = p
            .inner
            .iter()
            .map(|s| positions[s])
            .min()
            .unwrap_or(positions[&v]);
        let crosses_store = order[first..positions[&v]]
            .iter()
            .any(|&s| g.node(s).is_ok_and(|n| n.op() == Op::Store));
        if crosses_store {
            // Suppress partial fusions of the same expression as well.
            reserved.insert(v);
            reserved.extend(p.inner.iter().copied());
            continue;
        }
        if skip.contains(&v)
            || reserved.contains(&v)
            || p.inner.iter().any(|s| {
                reserved.contains(s)
                    || outputs.contains(s)
                    || skip.contains(s)
                    || consumers
                        .get(s)
                        .is_some_and(|cs| cs.iter().any(|c| *c != v && !p.inner.contains(c)))
            })
        {
            continue;
        }
        reserved.insert(v);
        reserved.extend(p.inner.iter().copied());
        plans.insert(v, p);
    }
    plans
}

/// Ordered max uses contiguous chunks and an ordered merge. A NaN resets the
/// serial ternary-max accumulator; track that reset to preserve NaNs and ties.
/// Add/multiply use strided lanes; floating-point association may change.
pub(super) fn warp_reduce(count: usize, op: ReduceOp, x: &str, acc: &str) -> String {
    match op {
        ReduceOp::Max => {
            let chunk = count.div_ceil(32);
            format!("float {acc}=-INFINITY; int {acc}_reset=0; for(size_t r=lane*{chunk};r<(lane+1)*{chunk} && r<{count};r++){{float next={x}; {acc}_reset|=(next!=next); {acc}=({acc}>next)?{acc}:next;}}\nfor(int offset=1;offset<32;offset*=2){{float next=__shfl_down_sync(0xffffffff,{acc},offset); int reset=__shfl_down_sync(0xffffffff,{acc}_reset,offset); if(lane%(2*offset)==0 && (lane+offset)*{chunk}<{count}){{if(reset || !({acc}>next)) {acc}=next; {acc}_reset|=reset;}}}}\n{acc}=__shfl_sync(0xffffffff,{acc},0);\n")
        }
        _ => {
            let (init, combine) = if op == ReduceOp::Add {
                ("0", "+")
            } else {
                ("1", "*")
            };
            format!("float {acc}={init}; for(size_t r=lane;r<{count};r+=32) {acc}={acc}{combine}({x}); for(int offset=16;offset>0;offset/=2) {acc}={acc}{combine}__shfl_down_sync(0xffffffff,{acc},offset); {acc}=__shfl_sync(0xffffffff,{acc},0);\n")
        }
    }
}
impl Fusion {
    pub fn code(&self, e: &mut Emitter<'_>, v: Value, name: &str) -> Result<String> {
        let (rows, width) = (self.rows, self.width);
        let mut code=format!("// fused row reduction\nconst size_t lane=threadIdx.x%32,row=(size_t)blockIdx.x*8+threadIdx.x/32; if(row>={rows})return;\n");
        match self.kind {
            Kind::Rms { sum } => {
                let x = e.read(e.g.node(sum)?.src()[0], &format!("r*{rows}+row"))?;
                code += &warp_reduce(width, ReduceOp::Add, &x, "row_sum");
                e.overrides.insert(sum, "row_sum".into());
                let y = e.read(v, "i")?;
                e.overrides.remove(&sum);
                code+=&format!("for(size_t col=lane;col<{width};col+=32){{size_t i=row*{width}+col; {name}[i]={y};}}\n");
            }
            Kind::Softmax { max, sum, exps } => {
                let x = e.read(e.g.node(max)?.src()[0], &format!("r*{rows}+row"))?;
                if width > 4096 {
                    // Stream wide rows rather than spilling a width/32 array
                    // per lane. The max itself is unobserved inside this fusion:
                    // any NaN still propagates through the exponential sum to
                    // every output, and signed-zero ties have identical exp2.
                    code += &format!("// streaming wide softmax\nfloat row_max=-INFINITY; for(size_t r=lane;r<{width};r+=32) row_max=fmaxf(row_max,({x}));\nfor(int offset=16;offset>0;offset/=2)row_max=fmaxf(row_max,__shfl_down_sync(0xffffffff,row_max,offset)); row_max=__shfl_sync(0xffffffff,row_max,0);\n");
                    e.overrides.insert(max, "row_max".into());
                    let exponent = e.read(exps, "i")?;
                    code += &format!("float row_sum=0; for(size_t col=lane;col<{width};col+=32){{size_t i=row*{width}+col;row_sum+=({exponent});}}\nfor(int offset=16;offset>0;offset/=2)row_sum+=__shfl_down_sync(0xffffffff,row_sum,offset); row_sum=__shfl_sync(0xffffffff,row_sum,0);\n");
                    e.overrides.insert(sum, "row_sum".into());
                    let y = e.read(v, "i")?;
                    e.overrides.remove(&max);
                    e.overrides.remove(&sum);
                    code += &format!("for(size_t col=lane;col<{width};col+=32){{size_t i=row*{width}+col;{name}[i]={y};}}\n");
                    return Ok(code);
                }
                code += &warp_reduce(width, ReduceOp::Max, &x, "row_max");
                e.overrides.insert(max, "row_max".into());
                let x = e.read(exps, "i")?;
                let chunks = width.div_ceil(32);
                code+=&format!("float row_values[{chunks}],row_sum=0;\n#pragma unroll\nfor(size_t q=0;q<{chunks};q++){{size_t col=lane+q*32,i=row*{width}+col; float value=(col<{width})?({x}):0; row_values[q]=value; row_sum+=value;}}\nfor(int offset=16;offset>0;offset/=2)row_sum+=__shfl_down_sync(0xffffffff,row_sum,offset); row_sum=__shfl_sync(0xffffffff,row_sum,0);\n");
                e.overrides.insert(sum, "row_sum".into());
                e.overrides.insert(exps, "row_values[q]".into());
                let y = e.read(v, "i")?;
                for r in [max, sum, exps] {
                    e.overrides.remove(&r);
                }
                code+=&format!("#pragma unroll\nfor(size_t q=0;q<{chunks};q++){{size_t col=lane+q*32,i=row*{width}+col; if(col<{width}) {name}[i]={y};}}\n");
            }
        }
        Ok(code)
    }
}
