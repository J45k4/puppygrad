use std::collections::HashMap;

use crate::graph_ir::{
    GraphBlock, GraphCallArg, GraphForIter, GraphForStmt, GraphFunction, GraphMatchCase,
    GraphModule, GraphStmt, Node, NodeId, NodeKind,
};
use crate::parser::{BinaryOp, DType, Dim, Literal, Pattern, TypeExpr, UnaryOp};

/// Generate a C source string from the lowered graph IR module.
pub fn generate_c_code(module: &GraphModule) -> String {
    let mut generator = CCodeGenerator::new();
    generator.generate_module(module)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CType {
    Int,
    Float,
    Bool,
    String,
    Void,
    Custom(String),
    Unknown,
}

impl CType {
    fn to_c_decl(&self) -> String {
        match self {
            CType::Int => "int64_t".to_string(),
            CType::Float => "double".to_string(),
            CType::Bool => "bool".to_string(),
            CType::String => "const char*".to_string(),
            CType::Void => "void".to_string(),
            CType::Custom(name) => name.clone(),
            CType::Unknown => "double".to_string(),
        }
    }

    fn merge(a: &CType, b: &CType) -> CType {
        if a == b {
            return a.clone();
        }
        if matches!(a, CType::Unknown) {
            return b.clone();
        }
        if matches!(b, CType::Unknown) {
            return a.clone();
        }
        match (a, b) {
            (CType::Float, _) | (_, CType::Float) => CType::Float,
            (CType::Int, CType::Int) => CType::Int,
            (CType::String, CType::String) => CType::String,
            (CType::Bool, CType::Bool) => CType::Bool,
            _ => CType::Unknown,
        }
    }
}

#[derive(Clone, Debug)]
struct CExpr {
    code: String,
    ty: CType,
}

struct ExprEmitter<'a> {
    nodes: &'a [Node],
    type_cache: HashMap<NodeId, CType>,
    expr_cache: HashMap<NodeId, String>,
    needs_string_header: bool,
}

impl<'a> ExprEmitter<'a> {
    fn new(nodes: &'a [Node]) -> Self {
        Self {
            nodes,
            type_cache: HashMap::new(),
            expr_cache: HashMap::new(),
            needs_string_header: false,
        }
    }

    fn needs_string_header(&self) -> bool {
        self.needs_string_header
    }

    fn expr(&mut self, node_id: NodeId) -> CExpr {
        let ty = self.node_type(node_id);
        if let Some(code) = self.expr_cache.get(&node_id) {
            return CExpr {
                code: code.clone(),
                ty,
            };
        }
        let code = self.compute_expr(node_id);
        self.expr_cache.insert(node_id, code.clone());
        CExpr { code, ty }
    }

    fn node_type(&mut self, node_id: NodeId) -> CType {
        if let Some(ty) = self.type_cache.get(&node_id) {
            return ty.clone();
        }

        let node = self
            .nodes
            .get(node_id.0)
            .unwrap_or_else(|| panic!("invalid node id {}", node_id.0));

        let ty = match &node.kind {
            NodeKind::Parameter { ty, .. } => ty
                .as_ref()
                .map(|ty| type_expr_to_ctype(ty))
                .unwrap_or(CType::Unknown),
            NodeKind::Literal(lit) => literal_ctype(lit),
            NodeKind::Binary { op, left, right } => {
                let left_ty = self.node_type(*left);
                let right_ty = self.node_type(*right);
                match op {
                    BinaryOp::Or | BinaryOp::And | BinaryOp::Eq | BinaryOp::NotEq => CType::Bool,
                    BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => CType::Bool,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                        combine_numeric_types(left_ty, right_ty)
                    }
                }
            }
            NodeKind::Unary { op, expr } => {
                let expr_ty = self.node_type(*expr);
                match op {
                    UnaryOp::Neg => match expr_ty {
                        CType::Float | CType::Int => expr_ty,
                        _ => CType::Unknown,
                    },
                    UnaryOp::Not => CType::Bool,
                }
            }
            NodeKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                let then_ty = self.node_type(*then_branch);
                let else_ty = self.node_type(*else_branch);
                CType::merge(&then_ty, &else_ty)
            }
            NodeKind::Match { cases, default, .. } => {
                let mut ty = self.node_type(*default);
                for case in cases {
                    let case_ty = self.node_type(case.value);
                    ty = CType::merge(&ty, &case_ty);
                }
                ty
            }
            NodeKind::Call { .. }
            | NodeKind::Field { .. }
            | NodeKind::MethodCall { .. }
            | NodeKind::TensorCtorShape(_)
            | NodeKind::TensorCtorValue(_)
            | NodeKind::Array(_)
            | NodeKind::Symbol { .. } => CType::Unknown,
            NodeKind::LoopVar { .. } => CType::Int,
        };

        self.type_cache.insert(node_id, ty.clone());
        ty
    }

    fn compute_expr(&mut self, node_id: NodeId) -> String {
        let node = self
            .nodes
            .get(node_id.0)
            .unwrap_or_else(|| panic!("invalid node id {}", node_id.0));
        match &node.kind {
            NodeKind::Parameter { name, .. } => name.clone(),
            NodeKind::Literal(lit) => literal_to_string(lit),
            NodeKind::Binary { op, left, right } => self.emit_binary(*op, *left, *right),
            NodeKind::Unary { op, expr } => {
                let expr = self.expr(*expr);
                let op_str = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                };
                format!("({}{})", op_str, expr.code)
            }
            NodeKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_expr = self.expr(*cond);
                let then_expr = self.expr(*then_branch);
                let else_expr = self.expr(*else_branch);
                format!(
                    "({} ? {} : {})",
                    cond_expr.code, then_expr.code, else_expr.code
                )
            }
            NodeKind::Match {
                scrutinee,
                cases,
                default,
            } => self.emit_match(*scrutinee, cases, *default),
            NodeKind::Call { callee, args } => {
                let callee_expr = self.expr(*callee);
                let args_code = self.emit_call_args(args);
                format!("{}({})", callee_expr.code, args_code)
            }
            NodeKind::Field { target, field } => {
                let target = self.expr(*target);
                format!("{}.{}", target.code, field)
            }
            NodeKind::MethodCall {
                target,
                method,
                args,
            } => {
                let target = self.expr(*target);
                let mut arg_list = vec![target.code];
                arg_list.extend(args.iter().map(|arg| self.emit_call_arg(arg)));
                format!("{}({})", method, arg_list.join(", "))
            }
            NodeKind::TensorCtorShape(dims) => {
                let dims_str = if dims.is_empty() {
                    String::new()
                } else {
                    dims.iter()
                        .map(dim_to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                format!("tensor_ctor({})", dims_str)
            }
            NodeKind::TensorCtorValue(value) => {
                let value_expr = self.expr(*value);
                format!("tensor_from_array({})", value_expr.code)
            }
            NodeKind::Array(items) => {
                let values = items
                    .iter()
                    .map(|item| self.expr(*item).code)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("(double[]){{{}}}", values)
            }
            NodeKind::Symbol { name } => name.clone(),
            NodeKind::LoopVar { name } => name.clone(),
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, left: NodeId, right: NodeId) -> String {
        let left_expr = self.expr(left);
        let right_expr = self.expr(right);
        match op {
            BinaryOp::Eq | BinaryOp::NotEq => {
                if left_expr.ty == CType::String || right_expr.ty == CType::String {
                    self.needs_string_header = true;
                    let cmp = format!("strcmp({}, {})", left_expr.code, right_expr.code);
                    if matches!(op, BinaryOp::Eq) {
                        format!("({} == 0)", cmp)
                    } else {
                        format!("({} != 0)", cmp)
                    }
                } else {
                    let op_str = match op {
                        BinaryOp::Eq => "==",
                        BinaryOp::NotEq => "!=",
                        _ => unreachable!(),
                    };
                    format!("({} {} {})", left_expr.code, op_str, right_expr.code)
                }
            }
            BinaryOp::Or => format!("({} || {})", left_expr.code, right_expr.code),
            BinaryOp::And => format!("({} && {})", left_expr.code, right_expr.code),
            BinaryOp::Lt => format!("({} < {})", left_expr.code, right_expr.code),
            BinaryOp::LtEq => format!("({} <= {})", left_expr.code, right_expr.code),
            BinaryOp::Gt => format!("({} > {})", left_expr.code, right_expr.code),
            BinaryOp::GtEq => format!("({} >= {})", left_expr.code, right_expr.code),
            BinaryOp::Add => format!("({} + {})", left_expr.code, right_expr.code),
            BinaryOp::Sub => format!("({} - {})", left_expr.code, right_expr.code),
            BinaryOp::Mul => format!("({} * {})", left_expr.code, right_expr.code),
            BinaryOp::Div => format!("({} / {})", left_expr.code, right_expr.code),
        }
    }

    fn emit_call_args(&mut self, args: &[GraphCallArg]) -> String {
        args.iter()
            .map(|arg| self.emit_call_arg(arg))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn emit_call_arg(&mut self, arg: &GraphCallArg) -> String {
        match arg {
            GraphCallArg::Positional(expr) => self.expr(*expr).code,
            GraphCallArg::Keyword { name, value } => {
                format!("/*{}=*/{}", name, self.expr(*value).code)
            }
        }
    }

    fn emit_match(
        &mut self,
        scrutinee: NodeId,
        cases: &[GraphMatchCase],
        default: NodeId,
    ) -> String {
        let scrutinee_expr = self.expr(scrutinee);
        let mut current = self.expr(default).code;
        for case in cases.iter().rev() {
            let value_expr = self.expr(case.value).code;
            let condition = self.pattern_condition(&scrutinee_expr.code, &case.pattern);
            current = format!("({} ? {} : {})", condition, value_expr, current);
        }
        format!("({})", current)
    }

    fn pattern_condition(&mut self, scrutinee_code: &str, pattern: &Pattern) -> String {
        match pattern {
            Pattern::Wildcard => "true".to_string(),
            Pattern::Literal(lit) => match lit {
                Literal::String(_) => {
                    self.needs_string_header = true;
                    format!(
                        "(strcmp({}, {}) == 0)",
                        scrutinee_code,
                        literal_to_string(lit)
                    )
                }
                _ => format!("({} == {})", scrutinee_code, literal_to_string(lit)),
            },
            Pattern::Ident(name) => format!("/* binding {} */ true", name),
            Pattern::Call { callee, .. } => format!("/* unsupported pattern {} */ false", callee),
        }
    }
}

fn combine_numeric_types(left: CType, right: CType) -> CType {
    if matches!(left, CType::Unknown) || matches!(right, CType::Unknown) {
        return CType::Unknown;
    }
    if matches!(left, CType::Float) || matches!(right, CType::Float) {
        return CType::Float;
    }
    if matches!(left, CType::Int) && matches!(right, CType::Int) {
        return CType::Int;
    }
    CType::Unknown
}

fn literal_ctype(literal: &Literal) -> CType {
    match literal {
        Literal::Int(_) => CType::Int,
        Literal::Float(_) => CType::Float,
        Literal::Bool(_) => CType::Bool,
        Literal::String(_) => CType::String,
    }
}

fn literal_to_string(literal: &Literal) -> String {
    match literal {
        Literal::Int(value) => value.to_string(),
        Literal::Float(value) => format!("{:?}", value),
        Literal::Bool(value) => value.to_string(),
        Literal::String(value) => format!("\"{}\"", escape_c_string(value)),
    }
}

fn dim_to_string(dim: &Dim) -> String {
    match dim {
        Dim::Int(value) => value.to_string(),
        Dim::Ident(name) => name.clone(),
    }
}

fn escape_c_string(input: &str) -> String {
    let mut escaped = String::new();
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => escaped.push_str(&format!("\\x{:02x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

fn type_expr_to_ctype(ty: &TypeExpr) -> CType {
    match ty {
        TypeExpr::Tensor { dtype, .. } => {
            let base = dtype
                .map(|d| dtype_to_c_primitive(d).to_string())
                .unwrap_or_else(|| "float".to_string());
            CType::Custom(format!("{}*", base))
        }
    }
}

fn dtype_to_c_primitive(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::BF16 => "uint16_t",
        DType::F32 => "float",
        DType::F64 => "double",
        DType::I32 => "int32_t",
        DType::I64 => "int64_t",
        DType::Bool => "bool",
    }
}

struct CCodeGenerator {
    needs_string_header: bool,
}

impl CCodeGenerator {
    fn new() -> Self {
        Self {
            needs_string_header: false,
        }
    }

    fn generate_module(&mut self, module: &GraphModule) -> String {
        let mut sections = Vec::new();

        if !module.body.statements.is_empty() {
            let mut emitter = ExprEmitter::new(&module.nodes);
            let body = self.emit_block(&mut emitter, &module.body, 4);
            self.needs_string_header |= emitter.needs_string_header();
            let mut section = String::from("void module_body(void) {\n");
            section.push_str(&body);
            section.push_str("}\n");
            sections.push(section);
        }

        for function in &module.functions {
            sections.push(self.generate_function(function));
        }

        let mut output = String::new();
        output.push_str("#include <stdint.h>\n");
        output.push_str("#include <stdbool.h>\n");
        if self.needs_string_header {
            output.push_str("#include <string.h>\n");
        }

        if sections.is_empty() {
            output.push('\n');
            return output;
        }

        output.push('\n');
        for (index, section) in sections.iter().enumerate() {
            output.push_str(section);
            if !section.ends_with('\n') {
                output.push('\n');
            }
            if index + 1 != sections.len() {
                output.push('\n');
            }
        }
        if !output.ends_with('\n') {
            output.push('\n');
        }
        output
    }

    fn generate_function(&mut self, function: &GraphFunction) -> String {
        let mut emitter = ExprEmitter::new(&function.nodes);
        let return_type = self.infer_return_type(&mut emitter, &function.body);
        let params = if function.params.is_empty() {
            "void".to_string()
        } else {
            function
                .params
                .iter()
                .map(|param| {
                    let ty = param
                        .ty
                        .as_ref()
                        .map(|ty| type_expr_to_ctype(ty))
                        .unwrap_or_else(|| emitter.node_type(param.node));
                    format!("{} {}", ty.to_c_decl(), param.name)
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let mut result = format!(
            "{} {}({}) {{\n",
            return_type.to_c_decl(),
            function.name,
            params
        );
        result.push_str(&self.emit_block(&mut emitter, &function.body, 4));
        result.push_str("}\n");
        self.needs_string_header |= emitter.needs_string_header();
        result
    }

    fn infer_return_type(&self, emitter: &mut ExprEmitter<'_>, body: &GraphBlock) -> CType {
        let mut saw_unknown = false;
        for stmt in &body.statements {
            if let GraphStmt::Return(node) = stmt {
                let ty = emitter.node_type(*node);
                if ty == CType::Unknown {
                    saw_unknown = true;
                } else {
                    return ty;
                }
            }
        }
        if saw_unknown {
            CType::Unknown
        } else {
            CType::Void
        }
    }

    fn emit_block(
        &mut self,
        emitter: &mut ExprEmitter<'_>,
        block: &GraphBlock,
        indent: usize,
    ) -> String {
        let mut output = String::new();
        for stmt in &block.statements {
            output.push_str(&self.emit_stmt(emitter, stmt, indent));
        }
        output
    }

    fn emit_stmt(
        &mut self,
        emitter: &mut ExprEmitter<'_>,
        stmt: &GraphStmt,
        indent: usize,
    ) -> String {
        let indent_str = " ".repeat(indent);
        match stmt {
            GraphStmt::Let { name, value } => {
                let expr = emitter.expr(*value);
                let mut ty = expr.ty.clone();
                if ty == CType::Void {
                    ty = CType::Unknown;
                }
                format!(
                    "{}{} {} = {};\n",
                    indent_str,
                    ty.to_c_decl(),
                    name,
                    expr.code
                )
            }
            GraphStmt::Assign { name, value } => {
                let expr = emitter.expr(*value);
                format!("{}{} = {};\n", indent_str, name, expr.code)
            }
            GraphStmt::Expr(node) => {
                let expr = emitter.expr(*node);
                format!("{}{};\n", indent_str, expr.code)
            }
            GraphStmt::For(for_stmt) => self.emit_for_stmt(emitter, for_stmt, indent),
            GraphStmt::Return(node) => {
                let expr = emitter.expr(*node);
                format!("{}return {};\n", indent_str, expr.code)
            }
        }
    }

    fn emit_for_stmt(
        &mut self,
        emitter: &mut ExprEmitter<'_>,
        for_stmt: &GraphForStmt,
        indent: usize,
    ) -> String {
        let indent_str = " ".repeat(indent);
        match &for_stmt.head.iter {
            GraphForIter::IntRange { start, end } => {
                let ty = emitter.node_type(for_stmt.head.binding_node).to_c_decl();
                let mut result = format!(
                    "{}for ({} {} = {}; {} < {}; ++{}) {{\n",
                    indent_str,
                    ty,
                    for_stmt.head.binding,
                    start,
                    for_stmt.head.binding,
                    end,
                    for_stmt.head.binding
                );
                result.push_str(&self.emit_block(emitter, &for_stmt.body, indent + 4));
                result.push_str(&format!("{}}}\n", indent_str));
                result
            }
            GraphForIter::RangeCall(node) => {
                if let Some((start, end, step)) = self.try_parse_range_call(emitter, *node) {
                    let ty = emitter.node_type(for_stmt.head.binding_node).to_c_decl();
                    let binding = &for_stmt.head.binding;
                    let increment = match step.as_deref() {
                        Some("1") | None => format!("++{}", binding),
                        Some("-1") => format!("--{}", binding),
                        Some(step_expr) => format!("{} += {}", binding, step_expr),
                    };
                    let mut result = format!(
                        "{}for ({} {} = {}; {} < {}; {}) {{\n",
                        indent_str, ty, binding, start, binding, end, increment
                    );
                    result.push_str(&self.emit_block(emitter, &for_stmt.body, indent + 4));
                    result.push_str(&format!("{}}}\n", indent_str));
                    result
                } else {
                    format!("{}/* unsupported range iteration */\n", indent_str)
                }
            }
            GraphForIter::TupleBinding(name) => {
                format!("{}/* tuple binding {} not supported */\n", indent_str, name)
            }
        }
    }

    fn try_parse_range_call(
        &mut self,
        emitter: &mut ExprEmitter<'_>,
        node: NodeId,
    ) -> Option<(String, String, Option<String>)> {
        let call_node = emitter
            .nodes
            .get(node.0)
            .and_then(|node| match &node.kind {
                NodeKind::Call { callee, args } => Some((callee, args)),
                _ => None,
            })?;

        let callee_name = emitter.expr(*call_node.0).code;
        if callee_name != "range" {
            return None;
        }

        let mut start = None;
        let mut end = None;
        let mut step = None;
        let mut positional = Vec::new();

        for arg in call_node.1.iter() {
            match arg {
                GraphCallArg::Positional(expr) => positional.push(emitter.expr(*expr).code),
                GraphCallArg::Keyword { name, value } => {
                    let value_code = emitter.expr(*value).code;
                    match name.as_str() {
                        "start" => start = Some(value_code),
                        "stop" | "end" => end = Some(value_code),
                        "step" => step = Some(value_code),
                        _ => {}
                    }
                }
            }
        }

        match positional.len() {
            1 => {
                start.get_or_insert_with(|| "0".to_string());
                end.get_or_insert_with(|| positional[0].clone());
            }
            2 => {
                start.get_or_insert_with(|| positional[0].clone());
                end.get_or_insert_with(|| positional[1].clone());
            }
            3 => {
                start.get_or_insert_with(|| positional[0].clone());
                end.get_or_insert_with(|| positional[1].clone());
                step.get_or_insert_with(|| positional[2].clone());
            }
            _ => {}
        }

        let start = start.unwrap_or_else(|| "0".to_string());
        let end = end?;
        Some((start, end, step))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_ir::lower_program;
    use crate::parser::{Block, Expr, FuncDecl, Param, Program, Stmt, TopLevelDecl};

    fn ident(name: &str) -> Expr {
        Expr::Ident(name.to_string())
    }

    fn literal_int(value: i64) -> Expr {
        Expr::Literal(Literal::Int(value))
    }

    #[test]
    fn generates_simple_function() {
        let func = FuncDecl {
            name: "add_one".to_string(),
            params: vec![Param {
                name: "x".to_string(),
                ty: None,
            }],
            body: Block {
                stmts: vec![
                    Stmt::Let {
                        name: "y".to_string(),
                        value: Expr::Binary {
                            op: BinaryOp::Add,
                            left: Box::new(ident("x")),
                            right: Box::new(literal_int(1)),
                        },
                    },
                    Stmt::Return(ident("y")),
                ],
            },
        };

        let program = Program {
            items: vec![TopLevelDecl::FuncDecl(func)],
        };
        let module = lower_program(&program);

        let code = generate_c_code(&module);
        let expected = "#include <stdint.h>\n#include <stdbool.h>\n\n".to_string()
            + "double add_one(double x) {\n"
            + "    double y = (x + 1);\n"
            + "    return (x + 1);\n"
            + "}\n";

        assert_eq!(code, expected);
    }
}
