//! Python-style graph construction. Functions and finite loops expand to Pops.
use super::expression::{self, Expr};
use super::pop::{Arg, DType, Error, Graph, Op, ReduceOp, Result, Scalar, Value};
use super::spec::{verify, Spec};
use std::collections::{HashMap, HashSet};

pub struct Program {
    pub graph: Graph,
    pub root: Value,
    pub bindings: Vec<Binding>,
    pub states: Vec<StateSpec>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateSpec {
    pub name: String,
    pub slot: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
}
pub struct Binding {
    pub name: String,
    pub line: usize,
    pub value: Value,
}
#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub slot: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
}
#[derive(Clone, Default)]
pub struct Context {
    pub tensors: HashMap<String, TensorSpec>,
    pub constants: HashMap<String, Scalar>,
}
#[derive(Clone)]
enum Stmt {
    Bind(String, Expr, usize),
    Assert(Expr, usize),
    For(String, Expr, Vec<Stmt>, usize),
    Return(Expr, usize),
    Output(Vec<Expr>, usize),
}
struct Function {
    parameters: Vec<String>,
    body: Vec<Stmt>,
    namespace: String,
    line: usize,
}
struct Line {
    text: String,
    indent: usize,
    number: usize,
}
#[derive(Default)]
struct Expansion {
    active: Vec<String>,
    cache: HashMap<(String, Vec<Value>), Value>,
    steps: usize,
}

fn identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
fn reserved(s: &str) -> bool {
    matches!(s, "def" | "return" | "output" | "import" | "for" | "in")
}
fn primitive(s: &str) -> Option<Op> {
    serde_json::from_value(serde_json::Value::String(s.to_ascii_uppercase())).ok()
}
fn lines(source: &str) -> Result<Vec<Line>> {
    let mut out = vec![];
    for (i, raw) in source.lines().enumerate() {
        let mut quote = None;
        let mut escaped = false;
        let mut end = raw.len();
        for (j, c) in raw.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            if quote.is_some() && c == '\\' {
                escaped = true;
                continue;
            }
            if Some(c) == quote {
                quote = None;
            } else if quote.is_none() && matches!(c, '\'' | '"') {
                quote = Some(c);
            } else if quote.is_none() && c == '#' {
                end = j;
                break;
            }
        }
        let raw = raw[..end].trim_end();
        if raw.is_empty() {
            continue;
        }
        let text = raw.trim_start();
        let indent = raw.len() - text.len();
        if raw[..indent].chars().any(|c| c != ' ') {
            return Err(Error(format!("{}: indentation must use spaces", i + 1)));
        }
        out.push(Line {
            text: text.into(),
            indent,
            number: i + 1,
        });
    }
    Ok(out)
}
fn parse_expr(text: &str, line: usize) -> Result<Expr> {
    expression::parse(text).map_err(|e| Error(format!("{line}: {e}")))
}
fn block(lines: &[Line], cursor: &mut usize, indent: usize, function: bool) -> Result<Vec<Stmt>> {
    let mut out = vec![];
    let mut returned = false;
    if *cursor < lines.len() && lines[*cursor].indent > 0 && lines[*cursor].indent < indent {
        return Err(Error(format!(
            "{}: blocks require four spaces per indentation level",
            lines[*cursor].number
        )));
    }
    while *cursor < lines.len() && lines[*cursor].indent >= indent {
        let line = &lines[*cursor];
        if line.indent != indent {
            return Err(Error(format!(
                "{}: blocks require four spaces per indentation level",
                line.number
            )));
        }
        if returned {
            return Err(Error(format!(
                "{}: return must be the final statement",
                line.number
            )));
        }
        let text = &line.text;
        let n = line.number;
        if text.starts_with("def ") || text.starts_with("import ") {
            break;
        }
        if let Some(tail) = text.strip_prefix("for ") {
            let (var, range) = tail
                .strip_suffix(':')
                .and_then(|s| s.split_once(" in "))
                .ok_or_else(|| Error(format!("{n}: expected for name in range(count):")))?;
            if !identifier(var) || reserved(var) {
                return Err(Error(format!("{n}: invalid loop variable")));
            }
            let Expr::Call(name, args) = parse_expr(range, n)? else {
                return Err(Error(format!("{n}: expected range(count)")));
            };
            if name != "range" || args.len() != 1 {
                return Err(Error(format!("{n}: expected range(count)")));
            }
            *cursor += 1;
            let body = block(lines, cursor, indent + 4, false)?;
            if body.is_empty() {
                return Err(Error(format!("{n}: loop requires a body")));
            }
            out.push(Stmt::For(var.into(), args[0].clone(), body, n));
            continue;
        }
        if let Some(tail) = text.strip_prefix("return ") {
            if !function {
                return Err(Error(format!(
                    "{n}: return is only allowed at function body level"
                )));
            }
            out.push(Stmt::Return(parse_expr(tail, n)?, n));
            returned = true;
        } else if let Some(tail) = text.strip_prefix("output ") {
            if indent != 0 {
                return Err(Error(format!(
                    "{n}: output is only allowed at module level"
                )));
            }
            let Expr::List(values) = parse_expr(&format!("[{tail}]"), n)? else {
                unreachable!()
            };
            out.push(Stmt::Output(values, n));
        } else if text.starts_with("assert_eq(") {
            out.push(Stmt::Assert(parse_expr(text, n)?, n));
        } else {
            let (name, expr) = text
                .split_once('=')
                .ok_or_else(|| Error(format!("{n}: expected name = expression")))?;
            let name = name.trim();
            if !identifier(name) || reserved(name) {
                return Err(Error(format!("{n}: invalid binding name")));
            }
            out.push(Stmt::Bind(name.into(), parse_expr(expr.trim(), n)?, n));
        }
        *cursor += 1;
    }
    Ok(out)
}
fn read_function(
    lines: &[Line],
    cursor: &mut usize,
    namespace: &str,
) -> Result<(String, Function)> {
    let header = &lines[*cursor];
    let n = header.number;
    let text = header
        .text
        .strip_prefix("def ")
        .unwrap()
        .strip_suffix(':')
        .ok_or_else(|| Error(format!("{n}: function declaration must end with ':'")))?;
    let Expr::Call(name, args) = parse_expr(text, n)? else {
        return Err(Error(format!("{n}: invalid function declaration")));
    };
    if !identifier(&name)
        || primitive(&name).is_some()
        || reserved(&name)
        || matches!(
            name.as_str(),
            "dim" | "assert_eq" | "state" | "input" | "weight" | "config" | "arange" | "grad"
        )
    {
        return Err(Error(format!(
            "{n}: invalid function name or primitive name collision"
        )));
    }
    let mut parameters = vec![];
    for arg in args {
        let Expr::Name(param) = arg else {
            return Err(Error(format!(
                "{n}: function parameters must be unique identifiers"
            )));
        };
        if !identifier(&param) || reserved(&param) || parameters.contains(&param) {
            return Err(Error(format!(
                "{n}: function parameters must be unique identifiers"
            )));
        }
        parameters.push(param);
    }
    *cursor += 1;
    let body = block(lines, cursor, 4, true)?;
    if !matches!(body.last(), Some(Stmt::Return(..))) {
        return Err(Error(format!(
            "{n}: function requires a final return value"
        )));
    }
    let mut declared: HashSet<_> = parameters.iter().cloned().collect();
    for stmt in &body {
        if let Stmt::Bind(name, _, line) = stmt {
            if !declared.insert(name.clone()) {
                return Err(Error(format!("{line}: binding {name:?} already exists")));
            }
        }
    }
    let full = if namespace.is_empty() {
        name
    } else {
        format!("{namespace}.{name}")
    };
    Ok((
        full,
        Function {
            parameters,
            body,
            namespace: namespace.into(),
            line: n,
        },
    ))
}
fn library(functions: &mut HashMap<String, Function>) -> Result<()> {
    let lines = lines(include_str!("../../stdlib/nn.pup"))?;
    let mut cursor = 0;
    while cursor < lines.len() {
        if lines[cursor].indent != 0 || !lines[cursor].text.starts_with("def ") {
            return Err(Error("nn library may only declare functions".into()));
        }
        let (name, f) = read_function(&lines, &mut cursor, "nn")?;
        functions.insert(name, f);
    }
    Ok(())
}
pub fn parse(source: &str) -> Result<Program> {
    parse_with_context(source, &Context::default())
}
// Reserve literal caller slots even when a state declaration appears first.
fn parameter_end(stmts: &[Stmt]) -> usize {
    fn expr_end(expr: &Expr) -> usize {
        match expr {
            Expr::Call(name, args) => {
                let own = if name.eq_ignore_ascii_case("param") {
                    match args.first() {
                        Some(Expr::Number(n)) => n
                            .parse::<usize>()
                            .ok()
                            .and_then(|n| n.checked_add(1))
                            .unwrap_or(0),
                        _ => 0,
                    }
                } else {
                    0
                };
                own.max(args.iter().map(expr_end).max().unwrap_or(0))
            }
            Expr::List(args) => args.iter().map(expr_end).max().unwrap_or(0),
            Expr::Unary(_, x) => expr_end(x),
            Expr::Binary(_, a, b) => expr_end(a).max(expr_end(b)),
            _ => 0,
        }
    }
    stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::Bind(_, e, _) | Stmt::Assert(e, _) | Stmt::Return(e, _) => expr_end(e),
            Stmt::Output(es, _) => es.iter().map(expr_end).max().unwrap_or(0),
            Stmt::For(_, count, body, _) => expr_end(count).max(parameter_end(body)),
        })
        .max()
        .unwrap_or(0)
}
pub fn parse_with_context(source: &str, context: &Context) -> Result<Program> {
    let lines = lines(source)?;
    let mut cursor = 0;
    let mut functions = HashMap::new();
    library(&mut functions)?;
    let mut statements = vec![];
    let mut output_seen = false;
    while cursor < lines.len() {
        let line = &lines[cursor];
        if line.indent != 0 {
            return Err(Error(format!(
                "{}: unexpected indentation outside function",
                line.number
            )));
        }
        if output_seen && !line.text.starts_with("output ") {
            return Err(Error(format!(
                "{}: bindings must precede output declarations",
                line.number
            )));
        }
        if line.text.starts_with("def ") {
            let (name, f) = read_function(&lines, &mut cursor, "")?;
            if functions.contains_key(&format!("nn.{name}")) {
                return Err(Error(format!(
                    "{}: function {name:?} is provided by the standard library",
                    line.number
                )));
            }
            if functions.insert(name.clone(), f).is_some() {
                return Err(Error(format!(
                    "{}: function {name:?} already exists",
                    line.number
                )));
            }
        } else if let Some(module) = line.text.strip_prefix("import ") {
            if module != "nn" {
                return Err(Error(format!(
                    "{}: unknown library {module:?}; only nn is bundled",
                    line.number
                )));
            }
            // Legacy spelling: the bundled library is already in scope.
            cursor += 1;
        } else {
            let parsed = block(&lines, &mut cursor, 0, false)?;
            for stmt in parsed {
                if output_seen && !matches!(stmt, Stmt::Output(..)) {
                    return Err(Error(format!(
                        "{}: bindings must precede output declarations",
                        stmt.line()
                    )));
                }
                output_seen |= matches!(stmt, Stmt::Output(..));
                statements.push(stmt);
            }
        }
    }
    let state_slot_start = parameter_end(&statements).max(
        functions
            .values()
            .map(|f| parameter_end(&f.body))
            .max()
            .unwrap_or(0),
    );
    let mut builder = Builder {
        state_slot_start,
        graph: Graph::new(),
        functions,
        context,
        expansion: Expansion::default(),
        states: vec![],
    };
    let mut names = HashMap::new();
    let mut bindings = vec![];
    let mut outputs = vec![];
    builder.statements(
        &statements,
        &mut names,
        "",
        false,
        Some(&mut bindings),
        &mut outputs,
    )?;
    if outputs.is_empty() {
        return Err(Error(format!(
            "{}: expected at least one output",
            source.lines().count().max(1)
        )));
    }
    if outputs
        .iter()
        .any(|&v| builder.graph.node(v).unwrap().shape().is_none())
    {
        return Err(Error(
            "outputs must be tensor values; attach writes through AFTER".into(),
        ));
    }
    let root = builder.graph.apply(Op::Sink, &outputs, Arg::None)?;
    verify(&builder.graph, root, Spec::Tensor)?;
    let reachable = builder
        .graph
        .toposort(root)?
        .into_iter()
        .collect::<HashSet<_>>();
    for index in 0..builder.graph.len() {
        // Writes must participate in the graph rather than disappear as unused bindings.
        let value = builder.graph.value_at(index);
        if builder.graph.node(value)?.op() == Op::Store && !reachable.contains(&value) {
            return Err(Error(
                "STORE must be attached to an output through AFTER".into(),
            ));
        }
    }
    Ok(Program {
        graph: builder.graph,
        root,
        bindings,
        states: builder.states,
    })
}
impl Stmt {
    fn line(&self) -> usize {
        match self {
            Self::Bind(_, _, n)
            | Self::Assert(_, n)
            | Self::For(_, _, _, n)
            | Self::Return(_, n)
            | Self::Output(_, n) => *n,
        }
    }
}
struct Builder<'a> {
    state_slot_start: usize,
    graph: Graph,
    functions: HashMap<String, Function>,
    context: &'a Context,
    expansion: Expansion,
    states: Vec<StateSpec>,
}
impl Builder<'_> {
    fn statements(
        &mut self,
        stmts: &[Stmt],
        names: &mut HashMap<String, Value>,
        namespace: &str,
        rebind: bool,
        mut bindings: Option<&mut Vec<Binding>>,
        outputs: &mut Vec<Value>,
    ) -> Result<Option<Value>> {
        for stmt in stmts {
            self.expansion.steps += 1;
            if self.expansion.steps > 100_000 {
                return Err(Error("graph expansion exceeds 100000 statements".into()));
            }
            let result = (|| -> Result<Option<Value>> {
                match stmt {
                    Stmt::Bind(name, expr, line) => {
                        if !rebind && names.contains_key(name) {
                            return Err(Error(format!("binding {name:?} already exists")));
                        }
                        let value = self.eval(expr, names, namespace)?;
                        names.insert(name.clone(), value);
                        if let Some(b) = bindings.as_deref_mut() {
                            b.push(Binding {
                                name: name.clone(),
                                line: *line,
                                value,
                            });
                        }
                    }
                    Stmt::Assert(expr, _) => {
                        self.eval(expr, names, namespace)?;
                    }
                    Stmt::Return(expr, _) => return self.eval(expr, names, namespace).map(Some),
                    Stmt::Output(exprs, _) => {
                        for expr in exprs {
                            outputs.push(self.eval(expr, names, namespace)?);
                        }
                    }
                    Stmt::For(var, count, body, _) => {
                        if names.contains_key(var) {
                            return Err(Error(format!(
                                "loop variable {var:?} shadows an existing binding"
                            )));
                        }
                        let count = self.integer(count, names, namespace)?;
                        if !(0..=4096).contains(&count) {
                            return Err(Error("range count must be between 0 and 4096".into()));
                        }
                        // Loop body names are SSA versions. Only existing bindings carry out.
                        let carried: Vec<_> = names.keys().cloned().collect();
                        let mut locals = names.clone();
                        for i in 0..count {
                            locals.insert(var.clone(), self.graph.constant(Scalar::Int(i))?);
                            self.statements(body, &mut locals, namespace, true, None, outputs)?;
                        }
                        for name in carried {
                            names.insert(name.clone(), locals[&name]);
                        }
                    }
                }
                Ok(None)
            })();
            let value = result.map_err(|e| Error(format!("{}: {e}", stmt.line())))?;
            if value.is_some() {
                return Ok(value);
            }
        }
        Ok(None)
    }
    fn integer(&mut self, expr: &Expr, names: &HashMap<String, Value>, ns: &str) -> Result<i64> {
        let value = self.eval(expr, names, ns)?;
        match self.graph.node(value)?.arg() {
            Arg::Scalar(Scalar::Int(n)) => Ok(*n),
            _ => Err(Error("expected a static integer constant".into())),
        }
    }
    fn dims(
        &mut self,
        expr: &Expr,
        names: &HashMap<String, Value>,
        ns: &str,
    ) -> Result<Vec<usize>> {
        let Expr::List(items) = expr else {
            return Err(Error("expected a dimension list".into()));
        };
        items
            .iter()
            .map(|x| {
                usize::try_from(self.integer(x, names, ns)?)
                    .map_err(|_| Error("dimension must be nonnegative".into()))
            })
            .collect()
    }
    fn text(&mut self, expr: &Expr, names: &HashMap<String, Value>, ns: &str) -> Result<String> {
        let Expr::Text(text, formatted) = expr else {
            return Err(Error("expected string literal".into()));
        };
        if !formatted {
            return Ok(text.clone());
        }
        let mut out = String::new();
        let mut rest = text.as_str();
        while let Some((before, tail)) = rest.split_once('{') {
            out.push_str(before);
            let (field, next) = tail
                .split_once('}')
                .ok_or_else(|| Error("unclosed format field".into()))?;
            if !identifier(field) {
                return Err(Error("format fields must name integer bindings".into()));
            }
            let n = self.integer(&Expr::Name(field.into()), names, ns)?;
            out.push_str(&n.to_string());
            rest = next;
        }
        out.push_str(rest);
        Ok(out)
    }
    fn eval(&mut self, expr: &Expr, names: &HashMap<String, Value>, ns: &str) -> Result<Value> {
        match expr {
            Expr::Name(n) if matches!(n.as_str(), "true" | "false") => {
                self.graph.constant(Scalar::Bool(n == "true"))
            }
            Expr::Name(n) => names.get(n).copied().ok_or_else(|| {
                Error(format!(
                    "unknown value {n:?}; sources must be defined first"
                ))
            }),
            Expr::Number(n) => {
                if !n.contains(['.', 'e', 'E']) {
                    return self.graph.constant(Scalar::Int(
                        n.parse().map_err(|_| {
                            Error("invalid or out-of-range integer constant".into())
                        })?,
                    ));
                }
                let v: f64 = n
                    .parse()
                    .map_err(|_| Error("invalid float constant".into()))?;
                if !v.is_finite() {
                    return Err(Error("float literal must be finite".into()));
                }
                self.graph.constant(Scalar::float(v))
            }
            Expr::Text(..) => Err(Error(
                "strings are only allowed in input, weight and config bindings".into(),
            )),
            Expr::List(items) => {
                let src = items
                    .iter()
                    .map(|x| self.eval(x, names, ns))
                    .collect::<Result<Vec<_>>>()?;
                if src.len() == 1 {
                    Ok(src[0])
                } else {
                    self.graph.apply(Op::Stack, &src, Arg::None)
                }
            }
            Expr::Unary(op, x) => {
                let x = self.eval(x, names, ns)?;
                if *op == '+' {
                    return Ok(x);
                }
                match self.graph.node(x)?.arg() {
                    Arg::Scalar(Scalar::Int(n)) => {
                        let n = n
                            .checked_neg()
                            .ok_or_else(|| Error("integer overflow".into()))?;
                        self.graph.constant(Scalar::Int(n))
                    }
                    Arg::Scalar(Scalar::Float(bits)) => {
                        let n = -f64::from_bits(*bits);
                        self.graph.constant(Scalar::float(n))
                    }
                    _ => self.graph.apply(Op::Neg, &[x], Arg::None),
                }
            }
            Expr::Binary(op, a, b) => {
                let a = self.eval(a, names, ns)?;
                let b = self.eval(b, names, ns)?;
                if let (Arg::Scalar(Scalar::Int(av)), Arg::Scalar(Scalar::Int(bv))) =
                    (self.graph.node(a)?.arg(), self.graph.node(b)?.arg())
                {
                    let (av, bv) = (*av, *bv);
                    let folded = match op.as_str() {
                        "+" => av.checked_add(bv),
                        "-" => av.checked_sub(bv),
                        "*" => av.checked_mul(bv),
                        "//" => av.checked_div(bv).and_then(|q| {
                            q.checked_sub(i64::from(av % bv != 0 && (av < 0) != (bv < 0)))
                        }),
                        "<" => return self.graph.constant(Scalar::Bool(av < bv)),
                        ">" => return self.graph.constant(Scalar::Bool(av > bv)),
                        "==" => return self.graph.constant(Scalar::Bool(av == bv)),
                        _ => None,
                    };
                    if op != "/" {
                        return self.graph.constant(Scalar::Int(folded.ok_or_else(|| {
                            Error("integer overflow or division by zero".into())
                        })?));
                    }
                }
                let opcode = match op.as_str() {
                    "+" => Op::Add,
                    "-" => Op::Sub,
                    "*" => Op::Mul,
                    "/" => Op::Fdiv,
                    "<" => Op::Cmplt,
                    ">" => return self.graph.apply(Op::Cmplt, &[b, a], Arg::None),
                    _ => {
                        return Err(Error(format!(
                            "{op} currently requires static integer operands"
                        )))
                    }
                };
                let (mut a, mut b) = (a, b);
                if opcode == Op::Fdiv {
                    if matches!(self.graph.node(a)?.dtype(), DType::I32 | DType::WeakInt) {
                        a = self.graph.cast(a, DType::F32)?;
                    }
                    if matches!(self.graph.node(b)?.dtype(), DType::I32 | DType::WeakInt) {
                        b = self.graph.cast(b, DType::F32)?;
                    }
                }
                self.graph.apply(opcode, &[a, b], Arg::None)
            }
            Expr::Call(name, args) => self.call(name, args, names, ns),
        }
    }
    fn call(
        &mut self,
        name: &str,
        args: &[Expr],
        names: &HashMap<String, Value>,
        ns: &str,
    ) -> Result<Value> {
        let mut qualified = if ns.is_empty() || name.contains('.') {
            name.into()
        } else {
            format!("{ns}.{name}")
        };
        // Resolve prelude names to the same function identity as legacy nn.* calls.
        if ns.is_empty()
            && !name.contains('.')
            && self.functions.contains_key(&format!("nn.{name}"))
        {
            qualified = format!("nn.{name}");
        }
        if let Some(f) = self.functions.get(&qualified) {
            if args.len() != f.parameters.len() {
                return Err(Error(format!(
                    "function {qualified} expects {} arguments, got {}",
                    f.parameters.len(),
                    args.len()
                )));
            }
            if self.expansion.active.contains(&qualified) {
                return Err(Error(format!(
                    "recursive function expansion is not supported: {qualified}"
                )));
            }
            if self.expansion.active.len() >= 128 {
                return Err(Error("function expansion depth exceeds 128".into()));
            }
            let (params, body, namespace, line) = (
                f.parameters.clone(),
                f.body.clone(),
                f.namespace.clone(),
                f.line,
            );
            let actuals = args
                .iter()
                .map(|x| self.eval(x, names, ns))
                .collect::<Result<Vec<_>>>()?;
            let key = (qualified.clone(), actuals.clone());
            if let Some(&v) = self.expansion.cache.get(&key) {
                return Ok(v);
            }
            let mut locals = params.into_iter().zip(actuals).collect();
            self.expansion.active.push(qualified.clone());
            let result = self.statements(&body, &mut locals, &namespace, false, None, &mut vec![]);
            self.expansion.active.pop();
            let value = result
                .map_err(|e| {
                    Error(format!(
                        "in {qualified} (definition line {line}): body line {e}"
                    ))
                })?
                .ok_or_else(|| Error("function did not return".into()))?;
            if !self
                .graph
                .toposort(value)?
                .iter()
                .any(|&v| self.graph.node(v).unwrap().op() == Op::Store)
            {
                self.expansion.cache.insert(key, value);
            }
            return Ok(value);
        }
        match (name, args) {
            ("state", [key, dt, dims]) => {
                let name = self.text(key, names, ns)?;
                let dtype = dtype(dt)?;
                let shape = self.dims(dims, names, ns)?;
                let existing = self.states.iter().find(|s| s.name == name).cloned();
                let slot = if let Some(s) = existing {
                    if s.dtype != dtype || s.shape != shape {
                        return Err(Error(format!("inconsistent state declaration {name:?}")));
                    }
                    s.slot
                } else {
                    let context_end = self
                        .context
                        .tensors
                        .values()
                        .map(|t| t.slot)
                        .max()
                        .map_or(0, |n| n + 1);
                    let graph_end = (0..self.graph.len())
                        .filter_map(|i| {
                            match self.graph.node(self.graph.value_at(i)).unwrap().arg() {
                                Arg::Param(p) => Some(p.slot + 1),
                                _ => None,
                            }
                        })
                        .max()
                        .unwrap_or(0);
                    let first = context_end.max(graph_end).max(self.state_slot_start);
                    let slot = first;
                    self.states.push(StateSpec {
                        name,
                        slot,
                        dtype,
                        shape: shape.clone(),
                    });
                    slot
                };
                let value = self
                    .graph
                    .state_param(slot, dtype, super::pop::numel(&shape)?)?;
                return self.graph.reshape(value, &shape);
            }
            ("grad", [loss, input]) => {
                let loss = self.eval(loss, names, ns)?;
                let input = self.eval(input, names, ns)?;
                let key = ("grad".to_owned(), vec![loss, input]);
                if let Some(&value) = self.expansion.cache.get(&key) {
                    return Ok(value);
                }
                let value = super::autodiff::gradients(&mut self.graph, loss, &[input])?[0];
                self.expansion.cache.insert(key, value);
                return Ok(value);
            }
            ("grad", _) => return Err(Error("grad expects (scalar_loss, input)".into())),
            ("input" | "weight", [key]) => {
                let key = self.text(key, names, ns)?;
                let spec = self
                    .context
                    .tensors
                    .get(&key)
                    .ok_or_else(|| Error(format!("no external tensor binding for {key:?}")))?;
                let value = self.graph.param(
                    spec.slot,
                    spec.dtype,
                    Some(super::pop::numel(&spec.shape)?),
                )?;
                return self.graph.reshape(value, &spec.shape);
            }
            ("config", [key]) => {
                let key = self.text(key, names, ns)?;
                let scalar = *self
                    .context
                    .constants
                    .get(&key)
                    .ok_or_else(|| Error(format!("missing config value {key:?}")))?;
                return self.graph.constant(scalar);
            }
            ("dim", [tensor, axis]) => {
                let tensor = self.eval(tensor, names, ns)?;
                let axis = self.integer(axis, names, ns)?;
                let shape = self
                    .graph
                    .node(tensor)?
                    .shape()
                    .ok_or_else(|| Error("dim requires a tensor".into()))?;
                let dimension = *shape
                    .get(usize::try_from(axis).unwrap_or(usize::MAX))
                    .ok_or_else(|| Error("dim axis out of bounds".into()))?;
                return self.graph.constant(Scalar::Int(
                    i64::try_from(dimension)
                        .map_err(|_| Error("dimension exceeds i64 subset".into()))?,
                ));
            }
            ("assert_eq", [a, b]) => {
                let a = self.integer(a, names, ns)?;
                let b = self.integer(b, names, ns)?;
                if a != b {
                    return Err(Error(format!("assert_eq failed: {a} != {b}")));
                }
                return self.graph.constant(Scalar::Int(a));
            }
            ("arange", [end]) => {
                let n = self.integer(end, names, ns)?;
                if !(0..=65536).contains(&n) {
                    return Err(Error("arange static length must be 0..65536".into()));
                }
                let mut src = vec![];
                for i in 0..n {
                    let v = self.graph.constant(Scalar::Int(i))?;
                    src.push(self.graph.cast(v, DType::I32)?);
                }
                if src.is_empty() {
                    let z = self.graph.constant(Scalar::Int(0))?;
                    let v = self.graph.cast(z, DType::I32)?;
                    let sh = self.graph.shape_value(&[0])?;
                    return self.graph.apply(Op::Expand, &[v, sh], Arg::None);
                }
                return self.graph.apply(Op::Stack, &src, Arg::None);
            }
            _ => (),
        }
        let op = primitive(name)
            .ok_or_else(|| Error(format!("unknown or unimplemented Pop or function {name:?}")))?;
        let (src, arg) = match (op, args) {
            (Op::Param, [slot, dt, size]) => {
                let slot = usize::try_from(self.integer(slot, names, ns)?)
                    .map_err(|_| Error("negative parameter slot".into()))?;
                let dt = dtype(dt)?;
                let size = if matches!(size,Expr::Name(n)if n=="scalar") {
                    None
                } else {
                    Some(
                        usize::try_from(self.integer(size, names, ns)?)
                            .map_err(|_| Error("negative parameter size".into()))?,
                    )
                };
                return self.graph.param(slot, dt, size);
            }
            (Op::Const, [x]) => return self.eval(x, names, ns),
            (Op::Cast, [x, dt]) => (vec![self.eval(x, names, ns)?], Arg::DType(dtype(dt)?)),
            (Op::Permute, [x, axes]) => (
                vec![self.eval(x, names, ns)?],
                Arg::Axes(self.dims(axes, names, ns)?),
            ),
            (Op::Flip, [x, Expr::List(flags)]) => {
                let flags = flags
                    .iter()
                    .map(|f| match f {
                        Expr::Name(s) if s == "true" => Ok(true),
                        Expr::Name(s) if s == "false" => Ok(false),
                        _ => Err(Error("expected bool flip flag".into())),
                    })
                    .collect::<Result<Vec<_>>>()?;
                (vec![self.eval(x, names, ns)?], Arg::Flip(flags))
            }
            (Op::Reduce, [x, Expr::Name(reduction), count]) => {
                let reduction: ReduceOp = serde_json::from_value(serde_json::Value::String(
                    reduction.to_ascii_uppercase(),
                ))
                .map_err(|_| Error("REDUCE operator must be add, mul or max".into()))?;
                let count = usize::try_from(self.integer(count, names, ns)?)
                    .map_err(|_| Error("negative reduction axis count".into()))?;
                (
                    vec![self.eval(x, names, ns)?],
                    Arg::Reduce {
                        op: reduction,
                        num_axes: count,
                    },
                )
            }
            (
                Op::Add
                | Op::Sub
                | Op::Fdiv
                | Op::Mul
                | Op::Max
                | Op::Cmplt
                | Op::Where
                | Op::Neg
                | Op::Exp2
                | Op::Log2
                | Op::Sqrt
                | Op::Sin
                | Op::Stack
                | Op::Sink
                | Op::Reshape
                | Op::Expand
                | Op::Pad
                | Op::Shrink
                | Op::Window
                | Op::Index
                | Op::Load
                | Op::Store
                | Op::After,
                _,
            ) => (
                args.iter()
                    .map(|x| self.eval(x, names, ns))
                    .collect::<Result<Vec<_>>>()?,
                Arg::None,
            ),
            _ => return Err(Error(format!("invalid arguments for {op:?}"))),
        };
        self.graph.apply(op, &src, arg)
    }
}
fn dtype(expr: &Expr) -> Result<DType> {
    match expr {
        Expr::Name(n) => match n.as_str() {
            "f32" => Ok(DType::F32),
            "i32" => Ok(DType::I32),
            "u8" => Ok(DType::U8),
            "bool" => Ok(DType::Bool),
            _ => Err(Error(format!("unsupported dtype {n:?}"))),
        },
        _ => Err(Error("expected dtype name".into())),
    }
}
