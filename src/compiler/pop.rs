//! Pop graph nodes implement a static subset of tinygrad's UOp contract,
//! pinned in third_party/tinygrad.
//! The node identity is (op, sources, argument). Shapes and dtypes are derived.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub(crate) String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
pub type Shape = Vec<usize>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Void,
    Bool,
    I32,
    U8,
    F32,
    WeakInt,
    WeakFloat,
}
impl DType {
    pub fn is_weak(self) -> bool {
        matches!(self, Self::WeakInt | Self::WeakFloat)
    }
}

/// CONST has a value-derived weak dtype (except Bool), just like the pinned spec.
/// IEEE f64 bits preserve signed zero and NaN payloads during structural sharing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Float(u64),
}
impl Scalar {
    pub fn float(value: f64) -> Self {
        Self::Float(value.to_bits())
    }
    pub fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::Int(_) => DType::WeakInt,
            Self::Float(_) => DType::WeakFloat,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Op {
    Param,
    Const,
    Stack,
    Cast,
    Add,
    Sub,
    Fdiv,
    Mul,
    Max,
    Neg,
    Exp2,
    Log2,
    Sqrt,
    Sin,
    Cmplt,
    Where,
    Index,
    Load,
    Store,
    After,
    Reshape,
    Expand,
    Permute,
    Pad,
    Shrink,
    Flip,
    Window,
    Reduce,
    Sink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

/// Initial PARAM subset: scalar or flat external tensor, without device ownership.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParamArg {
    pub slot: usize,
    pub dtype: DType,
    pub size: Option<usize>,
    #[serde(default)]
    pub writable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Arg {
    #[default]
    None,
    Scalar(Scalar),
    Param(ParamArg),
    DType(DType),
    Axes(Vec<usize>),
    Flip(Vec<bool>),
    Reduce {
        op: ReduceOp,
        num_axes: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Value {
    graph: u64,
    index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    op: Op,
    src: Vec<Value>,
    arg: Arg,
}

#[derive(Debug)]
pub struct Pop {
    key: Key,
    dtype: DType,
    shape: Option<Shape>,
}
impl Pop {
    pub fn op(&self) -> Op {
        self.key.op
    }
    pub fn src(&self) -> &[Value] {
        &self.key.src
    }
    pub fn arg(&self) -> &Arg {
        &self.key.arg
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// None denotes a statement such as SINK; Some([]) denotes a scalar value.
    pub fn shape(&self) -> Option<&[usize]> {
        self.shape.as_deref()
    }
}

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(0);

/// An immutable arena with hash-consing of values. STORE has unique identity;
/// AFTER records explicit ordering between reads and writes.
#[derive(Debug)]
pub struct Graph {
    id: u64,
    nodes: Vec<Pop>,
    interned: HashMap<Key, Value>,
    params: HashMap<usize, Value>,
}
impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
impl Graph {
    pub fn new() -> Self {
        Self {
            id: NEXT_GRAPH
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("graph identity space exhausted"),
            nodes: vec![],
            interned: HashMap::new(),
            params: HashMap::new(),
        }
    }
    pub fn node(&self, value: Value) -> Result<&Pop> {
        require(value.graph == self.id, "value belongs to a different graph")?;
        self.nodes
            .get(value.index)
            .ok_or_else(|| Error("invalid value index".into()))
    }
    pub(crate) fn value_at(&self, index: usize) -> Value {
        Value {
            graph: self.id,
            index,
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn apply(&mut self, op: Op, src: &[Value], arg: Arg) -> Result<Value> {
        let sources = src
            .iter()
            .map(|&v| self.node(v))
            .collect::<Result<Vec<_>>>()?;
        let (dtype, shape) = self.infer(op, &sources, &arg)?;
        let key = Key {
            op,
            src: src.to_vec(),
            arg,
        };
        if let Some(&value) = self.interned.get(&key).filter(|_| op != Op::Store) {
            return Ok(value);
        }
        if let Arg::Param(p) = &key.arg {
            require(
                !self.params.contains_key(&p.slot),
                "parameter slot already has a different declaration",
            )?;
        }
        let value = Value {
            graph: self.id,
            index: self.nodes.len(),
        };
        if let Arg::Param(p) = &key.arg {
            self.params.insert(p.slot, value);
        }
        if op != Op::Store {
            self.interned.insert(key.clone(), value);
        }
        self.nodes.push(Pop { key, dtype, shape });
        Ok(value)
    }
    pub fn param(&mut self, slot: usize, dtype: DType, size: Option<usize>) -> Result<Value> {
        self.apply(
            Op::Param,
            &[],
            Arg::Param(ParamArg {
                slot,
                dtype,
                size,
                writable: false,
            }),
        )
    }
    pub fn state_param(&mut self, slot: usize, dtype: DType, size: usize) -> Result<Value> {
        self.apply(
            Op::Param,
            &[],
            Arg::Param(ParamArg {
                slot,
                dtype,
                size: Some(size),
                writable: true,
            }),
        )
    }
    /// Stores address rows of a flat/reshaped state allocation. AFTER preserves
    /// storage identity while adding explicit dependencies on completed writes.
    pub fn writable_base(&self, mut value: Value) -> Result<Value> {
        loop {
            let node = self.node(value)?;
            match node.op() {
                Op::Reshape | Op::After => value = node.src()[0],
                Op::Param if matches!(node.arg(), Arg::Param(p) if p.writable) => return Ok(value),
                _ => return Err(Error("STORE destination must index a writable state buffer (optionally reshaped/ordered)".into())),
            }
        }
    }
    pub(crate) fn unique_indices(&self, value: Value) -> bool {
        let Ok(n) = self.node(value) else {
            return false;
        };
        if n.shape() == Some(&[]) {
            return true;
        }
        fn integer(g: &Graph, v: Value) -> Option<i64> {
            let n = g.node(v).ok()?;
            match n.arg() {
                Arg::Scalar(Scalar::Int(x)) => Some(*x),
                _ if n.op() == Op::Cast => integer(g, n.src()[0]).map(|x| {
                    if n.dtype() == DType::I32 {
                        x as i32 as i64
                    } else {
                        x
                    }
                }),
                _ => None,
            }
        }
        match n.op() {
            Op::Stack => {
                let values = n
                    .src()
                    .iter()
                    .map(|&v| integer(self, v))
                    .collect::<Option<Vec<_>>>();
                values.is_some_and(|v| v.iter().copied().collect::<HashSet<_>>().len() == v.len())
            }
            Op::Cast => {
                let input = self.node(n.src()[0]).unwrap();
                if input.dtype() == n.dtype() {
                    return self.unique_indices(n.src()[0]);
                }
                if n.dtype() == DType::I32 && input.op() == Op::Stack {
                    let values = input
                        .src()
                        .iter()
                        .map(|&v| integer(self, v).map(|x| x as i32))
                        .collect::<Option<Vec<_>>>();
                    return values.is_some_and(|v| {
                        v.iter().copied().collect::<HashSet<_>>().len() == v.len()
                    });
                }
                false
            }
            Op::Add | Op::Sub => {
                (self.node(n.src()[1]).is_ok_and(|n| n.shape() == Some(&[]))
                    && (n.dtype() == self.node(n.src()[0]).unwrap().dtype())
                    && self.unique_indices(n.src()[0]))
                    || (self.node(n.src()[0]).is_ok_and(|n| n.shape() == Some(&[]))
                        && (n.dtype() == self.node(n.src()[1]).unwrap().dtype())
                        && self.unique_indices(n.src()[1]))
            }
            _ => false,
        }
    }
    pub fn constant(&mut self, value: Scalar) -> Result<Value> {
        self.apply(Op::Const, &[], Arg::Scalar(value))
    }
    pub fn cast(&mut self, value: Value, dtype: DType) -> Result<Value> {
        self.apply(Op::Cast, &[value], Arg::DType(dtype))
    }
    pub fn shape_value(&mut self, shape: &[usize]) -> Result<Value> {
        let src = shape
            .iter()
            .map(|&d| {
                self.constant(Scalar::Int(
                    i64::try_from(d).map_err(|_| Error("dimension exceeds i64 subset".into()))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if src.len() == 1 {
            Ok(src[0])
        } else {
            self.apply(Op::Stack, &src, Arg::None)
        }
    }
    pub fn reshape(&mut self, value: Value, shape: &[usize]) -> Result<Value> {
        let shape = self.shape_value(shape)?;
        self.apply(Op::Reshape, &[value, shape], Arg::None)
    }

    /// Deterministic iterative traversal; node sharing and very deep DAGs are safe.
    pub fn toposort(&self, root: Value) -> Result<Vec<Value>> {
        self.node(root)?;
        let mut seen = HashSet::new();
        let mut stack = vec![(root, false)];
        let mut order = vec![];
        while let Some((value, finish)) = stack.pop() {
            if finish {
                order.push(value);
            } else if seen.insert(value) {
                stack.push((value, true));
                stack.extend(self.node(value)?.src().iter().rev().map(|&v| (v, false)));
            }
        }
        Ok(order)
    }
    pub fn dump(&self, root: Value) -> Result<String> {
        use fmt::Write;
        let mut names = HashMap::new();
        let mut out = String::new();
        for (i, value) in self.toposort(root)?.into_iter().enumerate() {
            let node = self.node(value)?;
            let args = node
                .src()
                .iter()
                .map(|v| format!("%{}", names[v]))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                out,
                "%{i}: {:?}{:?} = {:?}({args}) {:?}",
                node.dtype(),
                node.shape(),
                node.op(),
                node.arg()
            )
            .unwrap();
            names.insert(value, i);
        }
        Ok(out)
    }

    fn dimensions(&self, value: &Pop) -> Result<Shape> {
        let scalar_dim = |node: &Pop| match (node.op(), node.arg()) {
            (Op::Const, Arg::Scalar(Scalar::Int(n))) => {
                usize::try_from(*n).map_err(|_| Error("dimension must be nonnegative".into()))
            }
            _ => Err(Error(
                "symbolic shape expressions are not implemented yet; expected integer CONST".into(),
            )),
        };
        if value.op() == Op::Stack {
            value
                .src()
                .iter()
                .map(|&v| scalar_dim(self.node(v)?))
                .collect()
        } else {
            Ok(vec![scalar_dim(value)?])
        }
    }

    fn infer(&self, op: Op, src: &[&Pop], arg: &Arg) -> Result<(DType, Option<Shape>)> {
        let arity = match op {
            Op::Param | Op::Const => Some(0),
            Op::Stack | Op::Sink | Op::After => None,
            Op::Add
            | Op::Sub
            | Op::Fdiv
            | Op::Mul
            | Op::Max
            | Op::Cmplt
            | Op::Reshape
            | Op::Expand
            | Op::Index
            | Op::Store => Some(2),
            Op::Where | Op::Pad | Op::Shrink | Op::Window => Some(3),
            _ => Some(1),
        };
        if let Some(n) = arity {
            require(
                src.len() == n,
                format!("{op:?} expects {n} sources, got {}", src.len()),
            )?;
        }
        let plain = || {
            require(
                *arg == Arg::None,
                format!("{op:?} does not accept an argument"),
            )
        };
        let shape = |s: &Pop| {
            s.shape
                .clone()
                .ok_or_else(|| Error("operation requires a tensor value, got a statement".into()))
        };
        let result = match op {
            Op::Param => {
                let Arg::Param(p) = arg else {
                    return Err(Error("PARAM requires ParamArg".into()));
                };
                require(
                    matches!(p.dtype, DType::Bool | DType::I32 | DType::U8 | DType::F32),
                    "PARAM subset requires a concrete scalar dtype",
                )?;
                (p.dtype, Some(p.size.into_iter().collect()))
            }
            Op::Const => {
                let Arg::Scalar(value) = arg else {
                    return Err(Error("CONST requires a scalar value".into()));
                };
                (value.dtype(), Some(vec![]))
            }
            Op::Sink => {
                plain()?;
                (DType::Void, None)
            }
            Op::After => {
                plain()?;
                require(
                    src.len() >= 2,
                    "AFTER requires a value and write dependencies",
                )?;
                require(
                    src[1..].iter().all(|s| s.dtype == DType::Void),
                    "AFTER dependencies must be statements",
                )?;
                (src[0].dtype, Some(shape(src[0])?))
            }
            Op::Store => {
                plain()?;
                require(src[0].op() == Op::Index, "STORE destination must be INDEX")?;
                self.writable_base(src[0].src()[0])?;
                require(
                    src[0].dtype == src[1].dtype && src[0].shape() == src[1].shape(),
                    "STORE requires matching concrete dtype and shape",
                )?;
                (DType::Void, None)
            }
            Op::Stack => {
                plain()?;
                if src.is_empty() {
                    (DType::Void, Some(vec![]))
                } else {
                    let mut out = vec![src.len()];
                    let first = shape(src[0])?;
                    require(
                        src.iter().all(|s| s.shape() == Some(first.as_slice())),
                        "STACK sources must have matching shapes",
                    )?;
                    out.extend(first);
                    (promote(src)?, Some(out))
                }
            }
            Op::Cast => {
                let Arg::DType(dtype) = arg else {
                    return Err(Error("CAST requires a dtype argument".into()));
                };
                require(
                    matches!(dtype, DType::Bool | DType::I32 | DType::U8 | DType::F32),
                    "CAST subset requires a concrete scalar dtype",
                )?;
                require(src[0].dtype != DType::Void, "cannot cast void")?;
                (*dtype, Some(shape(src[0])?))
            }
            Op::Add | Op::Sub | Op::Fdiv | Op::Mul | Op::Max | Op::Cmplt | Op::Where => {
                plain()?;
                let dtype = if op == Op::Where {
                    require(src[0].dtype == DType::Bool, "WHERE condition must be bool")?;
                    promote(&src[1..])?
                } else {
                    promote(src)?
                };
                let shapes = src.iter().map(|s| shape(s)).collect::<Result<Vec<_>>>()?;
                if op == Op::Fdiv {
                    require(
                        matches!(dtype, DType::F32 | DType::WeakFloat),
                        "FDIV subset requires floating-point operands",
                    )?;
                }
                (
                    if op == Op::Cmplt { DType::Bool } else { dtype },
                    Some(broadcast(&shapes)?),
                )
            }
            Op::Neg | Op::Exp2 | Op::Log2 | Op::Sqrt | Op::Sin => {
                plain()?;
                if op != Op::Neg {
                    require(
                        matches!(src[0].dtype, DType::F32 | DType::WeakFloat),
                        "float operation requires floating-point input",
                    )?;
                }
                (src[0].dtype, Some(shape(src[0])?))
            }
            Op::Reshape | Op::Expand => {
                plain()?;
                let dims = self.dimensions(src[1])?;
                let input = shape(src[0])?;
                let out = if op == Op::Reshape {
                    require(
                        numel(&dims)? == numel(&input)?,
                        "RESHAPE must preserve element count",
                    )?;
                    dims
                } else {
                    [dims, input].concat()
                }; // tinygrad EXPAND prepends axes
                (src[0].dtype, Some(out))
            }
            Op::Permute => {
                let Arg::Axes(axes) = arg else {
                    return Err(Error("PERMUTE requires axis indices".into()));
                };
                let input = shape(src[0])?;
                let mut sorted = axes.clone();
                sorted.sort_unstable();
                require(
                    sorted == (0..input.len()).collect::<Vec<_>>(),
                    "PERMUTE axes must form a complete permutation",
                )?;
                (src[0].dtype, Some(axes.iter().map(|&a| input[a]).collect()))
            }
            Op::Window => {
                plain()?;
                let input = shape(src[0])?;
                let kernel = self.dimensions(src[1])?;
                let stride = self.dimensions(src[2])?;
                require(
                    input.len() == 4 && kernel.len() == 2 && stride.len() == 2,
                    "WINDOW requires NCHW input and two kernel/stride dimensions",
                )?;
                require(
                    kernel.iter().chain(&stride).all(|&n| n > 0),
                    "WINDOW kernel and stride must be positive",
                )?;
                require(
                    input[2] >= kernel[0] && input[3] >= kernel[1],
                    "WINDOW kernel exceeds input",
                )?;
                (
                    src[0].dtype,
                    Some(vec![
                        input[0],
                        input[1],
                        (input[2] - kernel[0]) / stride[0] + 1,
                        (input[3] - kernel[1]) / stride[1] + 1,
                        kernel[0],
                        kernel[1],
                    ]),
                )
            }
            Op::Pad | Op::Shrink => {
                plain()?;
                require(
                    src[1].shape() == src[2].shape(),
                    "PAD/SHRINK offset and size sources must have matching shapes",
                )?;
                let input = shape(src[0])?;
                let offsets = self.dimensions(src[1])?;
                let sizes = self.dimensions(src[2])?;
                require(
                    input.len() == offsets.len() && input.len() == sizes.len(),
                    "PAD/SHRINK rank mismatch",
                )?;
                for ((&old, &offset), &size) in input.iter().zip(&offsets).zip(&sizes) {
                    let (extent, limit) = if op == Op::Pad {
                        (old, size)
                    } else {
                        (size, old)
                    };
                    require(
                        offset.checked_add(extent).is_some_and(|end| end <= limit),
                        "PAD/SHRINK extent out of bounds",
                    )?;
                }
                (src[0].dtype, Some(sizes))
            }
            Op::Flip => {
                let Arg::Flip(axes) = arg else {
                    return Err(Error("FLIP requires bool flags".into()));
                };
                let input = shape(src[0])?;
                require(
                    axes.len() == input.len(),
                    "FLIP needs one flag per dimension",
                )?;
                (src[0].dtype, Some(input))
            }
            Op::Reduce => {
                let Arg::Reduce { num_axes, .. } = arg else {
                    return Err(Error("REDUCE requires (operator, num_axes)".into()));
                };
                let input = shape(src[0])?;
                require(*num_axes <= input.len(), "REDUCE axis count exceeds rank")?;
                (src[0].dtype, Some(input[*num_axes..].to_vec())) // leading axes
            }
            Op::Index => {
                plain()?;
                let base = shape(src[0])?;
                require(!base.is_empty(), "INDEX requires a non-scalar base")?;
                require(
                    matches!(src[1].dtype, DType::I32 | DType::WeakInt),
                    "INDEX requires integer indices",
                )?;
                let mut out = shape(src[1])?;
                out.extend_from_slice(&base[1..]);
                (src[0].dtype, Some(out))
            }
            Op::Load => {
                plain()?;
                require(src[0].op() == Op::Index, "LOAD subset requires INDEX")?;
                (src[0].dtype, Some(shape(src[0])?))
            }
        };
        if let Some(shape) = &result.1 {
            numel(shape)?;
        }
        Ok(result)
    }
}

pub(crate) fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
pub(crate) fn numel(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim)
            .ok_or_else(|| Error("element count overflow".into()))
    })
}
fn promote(src: &[&Pop]) -> Result<DType> {
    require(
        src.iter().all(|s| s.dtype != DType::Void),
        "ALU/STACK sources cannot be void",
    )?;
    let strong = src.iter().find(|s| !s.dtype.is_weak()).map(|s| s.dtype);
    let dtype = strong.unwrap_or(if src.iter().any(|s| s.dtype == DType::WeakFloat) {
        DType::WeakFloat
    } else {
        DType::WeakInt
    });
    require(
        src.iter().all(|s| s.dtype == dtype || s.dtype.is_weak()),
        "source dtypes must match; use CAST explicitly",
    )?;
    // Mixed weak float + concrete integer needs promotion/casts beyond this subset.
    require(
        !matches!(dtype, DType::Bool | DType::I32 | DType::U8)
            || src.iter().all(|s| s.dtype != DType::WeakFloat),
        "weak float with concrete integer/bool is not supported; use CAST",
    )?;
    // A weak integer promotes bool to integer upstream, requiring an explicit bool cast.
    require(
        dtype != DType::Bool || src.iter().all(|s| s.dtype == DType::Bool),
        "bool/integer mixing requires CAST",
    )?;
    Ok(dtype)
}
fn broadcast(shapes: &[Shape]) -> Result<Shape> {
    let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
    let mut output = vec![1; rank];
    for shape in shapes {
        for (out, &dim) in output[rank - shape.len()..].iter_mut().zip(shape) {
            require(
                *out == dim || *out == 1 || dim == 1,
                "incompatible broadcast dimensions",
            )?;
            if dim != 1 {
                *out = dim;
            }
        }
    }
    Ok(output)
}
