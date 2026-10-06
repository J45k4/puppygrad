//! Reverse-mode differentiation is graph construction, not a runtime tape.
//! Adjoint tensors use the same primitive Pops and backend as forward tensors.
use super::pop::{Arg, DType, Error, Graph, Op, ReduceOp, Result, Scalar, Value};
use std::collections::{HashMap, HashSet};

/// Differentiate a scalar f32 loss with respect to f32 graph values.
/// Unconnected inputs get zeros. Only paths leading to requested inputs are
/// differentiated; integer casts and comparisons stop those paths.
pub fn gradients(graph: &mut Graph, loss: Value, inputs: &[Value]) -> Result<Vec<Value>> {
    let node = graph.node(loss)?;
    if node.dtype() != DType::F32 || node.shape() != Some(&[][..]) {
        return Err(Error(
            "grad requires a scalar f32 loss; reduce tensor losses explicitly".into(),
        ));
    }
    for &input in inputs {
        if graph.node(input)?.dtype() != DType::F32 {
            return Err(Error("grad requires f32 differentiation targets".into()));
        }
    }
    let order = graph.toposort(loss)?;
    let mut needed: HashSet<_> = inputs.iter().copied().collect();
    for &v in &order {
        let n = graph.node(v)?;
        if n.dtype() == DType::F32 && n.src().iter().any(|s| needed.contains(s)) {
            needed.insert(v);
        }
    }
    let one = full(graph, &[], 1.)?;
    let mut reverse = Reverse {
        graph,
        needed,
        targets: inputs.iter().copied().collect(),
        adjoints: HashMap::from([(loss, one)]),
    };
    for v in order.into_iter().rev() {
        if !reverse.needed.contains(&v) {
            continue;
        }
        if let Some(&dy) = reverse.adjoints.get(&v) {
            reverse.propagate(v, dy)?;
        }
    }
    inputs
        .iter()
        .map(|&v| {
            if let Some(&gradient) = reverse.adjoints.get(&v) {
                Ok(gradient)
            } else {
                let shape = shape(reverse.graph, v)?;
                full(reverse.graph, &shape, 0.)
            }
        })
        .collect()
}

fn shape(g: &Graph, v: Value) -> Result<Vec<usize>> {
    g.node(v)?
        .shape()
        .map(<[usize]>::to_vec)
        .ok_or_else(|| Error("gradient requires a tensor value".into()))
}
fn unary(g: &mut Graph, op: Op, x: Value) -> Result<Value> {
    g.apply(op, &[x], Arg::None)
}
fn binary(g: &mut Graph, op: Op, x: Value, y: Value) -> Result<Value> {
    g.apply(op, &[x, y], Arg::None)
}
fn full(g: &mut Graph, dims: &[usize], x: f64) -> Result<Value> {
    let scalar = g.constant(Scalar::float(x))?;
    let scalar = g.cast(scalar, DType::F32)?;
    if dims.is_empty() {
        return Ok(scalar);
    }
    let dims = g.shape_value(dims)?;
    g.apply(Op::Expand, &[scalar, dims], Arg::None)
}
fn permute(g: &mut Graph, x: Value, axes: &[usize]) -> Result<Value> {
    if axes.iter().copied().eq(0..axes.len()) {
        return Ok(x);
    }
    g.apply(Op::Permute, &[x], Arg::Axes(axes.to_vec()))
}
fn reduce(g: &mut Graph, x: Value, op: ReduceOp, axes: usize) -> Result<Value> {
    if axes == 0 {
        return Ok(x);
    }
    g.apply(Op::Reduce, &[x], Arg::Reduce { op, num_axes: axes })
}
fn expand_leading(g: &mut Graph, x: Value, dims: &[usize]) -> Result<Value> {
    if dims.is_empty() {
        return Ok(x);
    }
    let dims = g.shape_value(dims)?;
    g.apply(Op::Expand, &[x, dims], Arg::None)
}
/// Undo NumPy-style elementwise broadcasting, including singleton and empty axes.
fn sum_to(g: &mut Graph, x: Value, target: &[usize]) -> Result<Value> {
    let old = shape(g, x)?;
    if old == target {
        return Ok(x);
    }
    if target.len() > old.len() {
        return Err(Error("invalid gradient broadcast rank".into()));
    }
    let lead = old.len() - target.len();
    let mut axes = Vec::new();
    let mut kept = Vec::new();
    for (i, &d) in old.iter().enumerate() {
        if i < lead || target[i - lead] == 1 && d != 1 {
            axes.push(i);
        } else {
            if d != target[i - lead] {
                return Err(Error("invalid gradient broadcast shape".into()));
            }
            kept.push(i);
        }
    }
    let count = axes.len();
    axes.extend(kept);
    let x = permute(g, x, &axes)?;
    let x = reduce(g, x, ReduceOp::Add, count)?;
    g.reshape(x, target)
}
fn select(g: &mut Graph, condition: Value, yes: Value, no: Value) -> Result<Value> {
    g.apply(Op::Where, &[condition, yes, no], Arg::None)
}
/// Matrix contractions remain ordinary reshape/multiply/reduce Pops.
fn matmul(g: &mut Graph, a: Value, b: Value) -> Result<Value> {
    let ash = shape(g, a)?;
    let bsh = shape(g, b)?;
    if ash.len() == 2 && bsh.len() == 2 {
        return super::tensor::matmul(g, a, b);
    }
    if ash.len() != 3 || bsh.len() != 3 || ash[2] != bsh[1] {
        return Err(Error("invalid batched gradient contraction".into()));
    }
    let a = g.reshape(a, &[ash[0], ash[1], 1, ash[2]])?;
    let b = permute(g, b, &[0, 2, 1])?;
    let b = g.reshape(b, &[bsh[0], 1, bsh[2], bsh[1]])?;
    let product = binary(g, Op::Mul, a, b)?;
    let product = permute(g, product, &[3, 0, 1, 2])?;
    reduce(g, product, ReduceOp::Add, 1)
}

struct Reverse<'a> {
    graph: &'a mut Graph,
    needed: HashSet<Value>,
    targets: HashSet<Value>,
    adjoints: HashMap<Value, Value>,
}
impl Reverse<'_> {
    fn wants(&self, v: Value) -> bool {
        self.needed.contains(&v) && self.graph.node(v).is_ok_and(|n| n.dtype() == DType::F32)
    }
    fn add(&mut self, v: Value, contribution: Value) -> Result<()> {
        if !self.wants(v) {
            return Ok(());
        }
        let target = shape(self.graph, v)?;
        let contribution = sum_to(self.graph, contribution, &target)?;
        let value = if let Some(&old) = self.adjoints.get(&v) {
            binary(self.graph, Op::Add, old, contribution)?
        } else {
            contribution
        };
        self.adjoints.insert(v, value);
        Ok(())
    }
    /// Recognize the library's contraction decomposition so its pullback also
    /// consists of contractions, rather than constructing a broadcast Jacobian.
    fn contraction(&mut self, v: Value, dy: Value) -> Result<bool> {
        let g = &mut self.graph;
        let n = g.node(v)?;
        if n.op() != Op::Reduce
            || n.arg()
                != &(Arg::Reduce {
                    op: ReduceOp::Add,
                    num_axes: 1,
                })
        {
            return Ok(false);
        }
        let permutation = n.src()[0];
        let p = g.node(permutation)?;
        if p.op() != Op::Permute {
            return Ok(false);
        }
        let product = p.src()[0];
        // A requested internal value is an observable differentiation boundary.
        // Use primitive pullbacks when bypassing it would lose its adjoint.
        if self.targets.contains(&permutation) || self.targets.contains(&product) {
            return Ok(false);
        }
        let mul = g.node(product)?;
        if mul.op() != Op::Mul {
            return Ok(false);
        }
        let dims = mul.shape().unwrap().to_vec();
        let rank = dims.len();
        if !(rank == 3 || rank == 4)
            || p.arg() != &Arg::Axes(std::iter::once(rank - 1).chain(0..rank - 1).collect())
        {
            return Ok(false);
        }
        let (a, b) = (mul.src()[0], mul.src()[1]);
        let ash = shape(g, a)?;
        let bsh = shape(g, b)?;
        if ash.len() != rank
            || bsh.len() != rank
            || ash[rank - 2] != 1
            || bsh[rank - 3] != 1
            || ash[rank - 1] != bsh[rank - 1]
            || g.node(a)?.dtype() != DType::F32
            || g.node(b)?.dtype() != DType::F32
        {
            return Ok(false);
        }
        let (m, n, k) = (dims[rank - 3], dims[rank - 2], dims[rank - 1]);
        let a_shape = if rank == 3 {
            vec![m, k]
        } else {
            vec![ash[0], m, k]
        };
        let bt_shape = if rank == 3 {
            vec![n, k]
        } else {
            vec![bsh[0], n, k]
        };
        let axes = if rank == 3 { vec![1, 0] } else { vec![0, 2, 1] };
        // Differentiate through the library's input views as one rule, while
        // preserving every explicitly requested intermediate boundary. Writing
        // dB=A.T@dY directly lets an SGD epilogue fuse into that contraction;
        // materializing (dY.T@A).T would otherwise need a full weight-sized arena.
        let left_input = if !self.targets.contains(&a)
            && g.node(a)?.op() == Op::Reshape
            && shape(g, g.node(a)?.src()[0])? == a_shape
        {
            Some(g.node(a)?.src()[0])
        } else {
            None
        };
        let right_view = if g.node(b)?.op() == Op::Reshape {
            Some(g.node(b)?.src()[0])
        } else {
            None
        };
        let right_input = if let Some(view) = right_view {
            let node = g.node(view)?;
            if !self.targets.contains(&b)
                && !self.targets.contains(&view)
                && node.op() == Op::Permute
                && node.arg() == &Arg::Axes(axes.clone())
                && shape(g, view)? == bt_shape
            {
                Some(node.src()[0])
            } else {
                None
            }
        } else {
            None
        };
        let a2 = if let Some(input) = left_input {
            input
        } else {
            g.reshape(a, &a_shape)?
        };
        let bt = g.reshape(b, &bt_shape)?;
        if self.wants(a) {
            let da = matmul(self.graph, dy, bt)?;
            if let Some(input) = left_input {
                self.add(input, da)?;
            } else {
                let sh = if rank == 3 {
                    vec![m, 1, k]
                } else {
                    vec![dims[0], m, 1, k]
                };
                let da = self.graph.reshape(da, &sh)?;
                self.add(a, da)?;
            }
        }
        if self.wants(b) {
            if let Some(input) = right_input {
                let a_t = permute(self.graph, a2, &axes)?;
                let db = matmul(self.graph, a_t, dy)?;
                self.add(input, db)?;
            } else {
                let dy_t = permute(self.graph, dy, &axes)?;
                let db = matmul(self.graph, dy_t, a2)?;
                let sh = if rank == 3 {
                    vec![1, n, k]
                } else {
                    vec![dims[0], 1, n, k]
                };
                let db = self.graph.reshape(db, &sh)?;
                self.add(b, db)?;
            }
        }
        Ok(true)
    }
    fn propagate(&mut self, v: Value, dy: Value) -> Result<()> {
        let node = self.graph.node(v)?;
        let (op, src, arg, dims) = (
            node.op(),
            node.src().to_vec(),
            node.arg().clone(),
            shape(self.graph, v)?,
        );
        if !src.iter().any(|&s| self.wants(s)) {
            return Ok(());
        }
        if self.contraction(v, dy)? {
            return Ok(());
        }
        let zero = full(self.graph, &[], 0.)?;
        let one = full(self.graph, &[], 1.)?;
        match op {
            Op::Param | Op::Const | Op::Cmplt => (),
            Op::Cast | Op::Load => {
                self.add(src[0], dy)?;
            }
            Op::Add => {
                self.add(src[0], dy)?;
                self.add(src[1], dy)?;
            }
            Op::Sub => {
                self.add(src[0], dy)?;
                let negative = unary(self.graph, Op::Neg, dy)?;
                self.add(src[1], negative)?;
            }
            Op::Mul => {
                for (input, other) in [(src[0], src[1]), (src[1], src[0])] {
                    if self.wants(input) {
                        let dx = binary(self.graph, Op::Mul, dy, other)?;
                        self.add(input, dx)?;
                    }
                }
            }
            Op::Fdiv => {
                if self.wants(src[0]) {
                    let dx = binary(self.graph, Op::Fdiv, dy, src[1])?;
                    self.add(src[0], dx)?;
                }
                if self.wants(src[1]) {
                    let dy_y = binary(self.graph, Op::Mul, dy, v)?;
                    let dx = binary(self.graph, Op::Fdiv, dy_y, src[1])?;
                    let dx = unary(self.graph, Op::Neg, dx)?;
                    self.add(src[1], dx)?;
                }
            }
            Op::Neg => {
                let dx = unary(self.graph, Op::Neg, dy)?;
                self.add(src[0], dx)?;
            }
            Op::Exp2 => {
                let ln2 = full(self.graph, &[], std::f64::consts::LN_2)?;
                let dx = binary(self.graph, Op::Mul, dy, v)?;
                let dx = binary(self.graph, Op::Mul, dx, ln2)?;
                self.add(src[0], dx)?;
            }
            Op::Log2 => {
                let ln2 = full(self.graph, &[], std::f64::consts::LN_2)?;
                let denom = binary(self.graph, Op::Mul, src[0], ln2)?;
                let dx = binary(self.graph, Op::Fdiv, dy, denom)?;
                self.add(src[0], dx)?;
            }
            Op::Sqrt => {
                let two = full(self.graph, &[], 2.)?;
                let denom = binary(self.graph, Op::Mul, two, v)?;
                let dx = binary(self.graph, Op::Fdiv, dy, denom)?;
                self.add(src[0], dx)?;
            }
            Op::Sin => {
                let half_pi = full(self.graph, &[], std::f64::consts::FRAC_PI_2)?;
                let shifted = binary(self.graph, Op::Add, src[0], half_pi)?;
                let cosine = unary(self.graph, Op::Sin, shifted)?;
                let dx = binary(self.graph, Op::Mul, dy, cosine)?;
                self.add(src[0], dx)?;
            }
            Op::Window => return Err(Error("grad through WINDOW requires overlapping scatter-add, which is not implemented yet".into())),
            Op::Max => {
                // Forward MAX selects the right operand at a tie (ReLU'(0)=0).
                let left = binary(self.graph, Op::Cmplt, src[1], src[0])?;
                let da = select(self.graph, left, dy, zero)?;
                let db = select(self.graph, left, zero, dy)?;
                self.add(src[0], da)?;
                self.add(src[1], db)?;
            }
            Op::Where => {
                let da = select(self.graph, src[0], dy, zero)?;
                let db = select(self.graph, src[0], zero, dy)?;
                self.add(src[1], da)?;
                self.add(src[2], db)?;
            }
            Op::Reshape => {
                let sh = shape(self.graph, src[0])?;
                let dx = self.graph.reshape(dy, &sh)?;
                self.add(src[0], dx)?;
            }
            Op::Expand => {
                let count = dims.len() - shape(self.graph, src[0])?.len();
                let dx = reduce(self.graph, dy, ReduceOp::Add, count)?;
                self.add(src[0], dx)?;
            }
            Op::Permute => {
                let Arg::Axes(axes) = arg else { unreachable!() };
                let mut inverse = vec![0; axes.len()];
                for (i, j) in axes.into_iter().enumerate() {
                    inverse[j] = i;
                }
                let dx = permute(self.graph, dy, &inverse)?;
                self.add(src[0], dx)?;
            }
            Op::Flip => {
                let dx = self.graph.apply(Op::Flip, &[dy], arg)?;
                self.add(src[0], dx)?;
            }
            Op::Pad | Op::Shrink => {
                let sh = shape(self.graph, src[0])?;
                let sh = self.graph.shape_value(&sh)?;
                let reverse = if op == Op::Pad { Op::Shrink } else { Op::Pad };
                let dx = self.graph.apply(reverse, &[dy, src[1], sh], Arg::None)?;
                self.add(src[0], dx)?;
            }
            Op::Stack => {
                for (i, &input) in src.iter().enumerate() {
                    if !self.wants(input) {
                        continue;
                    }
                    let sh = shape(self.graph, input)?;
                    let mut offsets = vec![0; sh.len() + 1];
                    offsets[0] = i;
                    let mut sizes = vec![1];
                    sizes.extend_from_slice(&sh);
                    let offsets = self.graph.shape_value(&offsets)?;
                    let sizes = self.graph.shape_value(&sizes)?;
                    let part = self
                        .graph
                        .apply(Op::Shrink, &[dy, offsets, sizes], Arg::None)?;
                    let dx = self.graph.reshape(part, &sh)?;
                    self.add(input, dx)?;
                }
            }
            Op::Reduce => {
                let Arg::Reduce { op: kind, num_axes } = arg else {
                    unreachable!()
                };
                let input_shape = shape(self.graph, src[0])?;
                let expanded = expand_leading(self.graph, dy, &input_shape[..num_axes])?;
                let dx = match kind {
                    ReduceOp::Add => expanded,
                    ReduceOp::Max => {
                        // Share the subgradient evenly among equal finite maxima.
                        let maximum = expand_leading(self.graph, v, &input_shape[..num_axes])?;
                        let smaller = binary(self.graph, Op::Cmplt, src[0], maximum)?;
                        let mask = select(self.graph, smaller, zero, one)?;
                        let count = reduce(self.graph, mask, ReduceOp::Add, num_axes)?;
                        let count = expand_leading(self.graph, count, &input_shape[..num_axes])?;
                        let selected = binary(self.graph, Op::Mul, expanded, mask)?;
                        binary(self.graph, Op::Fdiv, selected, count)?
                    }
                    ReduceOp::Mul => {
                        // Product rule without dividing by zero: count zeros and
                        // multiply the nonzero elements before selecting a case.
                        let below = binary(self.graph, Op::Cmplt, src[0], zero)?;
                        let above = binary(self.graph, Op::Cmplt, zero, src[0])?;
                        let nonpositive = select(self.graph, above, zero, one)?;
                        let zeros = select(self.graph, below, zero, nonpositive)?;
                        let is_zero = binary(self.graph, Op::Cmplt, zero, zeros)?;
                        let safe = select(self.graph, is_zero, one, src[0])?;
                        let product = reduce(self.graph, safe, ReduceOp::Mul, num_axes)?;
                        let product =
                            expand_leading(self.graph, product, &input_shape[..num_axes])?;
                        let count = reduce(self.graph, zeros, ReduceOp::Add, num_axes)?;
                        let count = expand_leading(self.graph, count, &input_shape[..num_axes])?;
                        let none = binary(self.graph, Op::Cmplt, count, one)?;
                        let two = full(self.graph, &[], 2.)?;
                        let at_most_one = binary(self.graph, Op::Cmplt, count, two)?;
                        let one_case = select(self.graph, at_most_one, zeros, zero)?;
                        let mask = select(self.graph, none, one, one_case)?;
                        let dx = binary(self.graph, Op::Fdiv, product, safe)?;
                        let dx = binary(self.graph, Op::Mul, dx, mask)?;
                        binary(self.graph, Op::Mul, expanded, dx)?
                    }
                };
                self.add(src[0], dx)?;
            }
            Op::Index => {
                return Err(Error(
                    "grad through INDEX requires scatter-add, which is not implemented yet".into(),
                ))
            }
            Op::Sink | Op::Store | Op::After => return Err(Error("cannot differentiate effectful state operations".into())),
        }
        Ok(())
    }
}
