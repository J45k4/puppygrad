#![allow(dead_code)]

use std::collections::HashMap;

use crate::parser::{
    Attr, BinaryOp, Block, CallArg, CallExpr, Dim, Expr, ForIter, ForStmt, FuncDecl, Literal,
    Pattern, Program, Stmt, TensorCtor, TopLevelDecl, TypeExpr, UnaryOp,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub enum GraphCallArg {
    Positional(NodeId),
    Keyword { name: String, value: NodeId },
}

#[derive(Debug, Clone)]
pub struct GraphMatchCase {
    pub pattern: Pattern,
    pub value: NodeId,
}

#[derive(Debug, Clone)]
pub enum GraphForIter {
    RangeCall(NodeId),
    IntRange { start: i64, end: i64 },
    TupleBinding(String),
}

#[derive(Debug, Clone)]
pub struct GraphForHead {
    pub binding: String,
    pub iter: GraphForIter,
    pub binding_node: NodeId,
}

#[derive(Debug, Clone)]
pub struct GraphForStmt {
    pub head: GraphForHead,
    pub body: GraphBlock,
}

#[derive(Debug, Clone)]
pub enum GraphStmt {
    Let { name: String, value: NodeId },
    Assign { name: String, value: NodeId },
    Expr(NodeId),
    For(GraphForStmt),
    Return(NodeId),
}

#[derive(Debug, Clone)]
pub struct GraphBlock {
    pub statements: Vec<GraphStmt>,
}

#[derive(Debug, Clone)]
pub struct GraphParam {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub node: NodeId,
}

#[derive(Debug, Clone)]
pub struct GraphTensorDecl {
    pub name: String,
    pub shape: Vec<Dim>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct GraphFunction {
    pub name: String,
    pub params: Vec<GraphParam>,
    pub body: GraphBlock,
    pub nodes: Vec<Node>,
    pub exported: bool,
}

#[derive(Debug, Clone)]
pub struct GraphModule {
    pub tensors: Vec<GraphTensorDecl>,
    pub functions: Vec<GraphFunction>,
    pub body: GraphBlock,
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
}

#[derive(Debug, Clone)]
pub enum NodeKind {
    Parameter {
        name: String,
        ty: Option<TypeExpr>,
    },
    Literal(Literal),
    Binary {
        op: BinaryOp,
        left: NodeId,
        right: NodeId,
    },
    Unary {
        op: UnaryOp,
        expr: NodeId,
    },
    If {
        cond: NodeId,
        then_branch: NodeId,
        else_branch: NodeId,
    },
    Match {
        scrutinee: NodeId,
        cases: Vec<GraphMatchCase>,
        default: NodeId,
    },
    Call {
        callee: NodeId,
        args: Vec<GraphCallArg>,
    },
    Field {
        target: NodeId,
        field: String,
    },
    MethodCall {
        target: NodeId,
        method: String,
        args: Vec<GraphCallArg>,
    },
    TensorCtorShape(Vec<Dim>),
    TensorCtorValue(NodeId),
    Array(Vec<NodeId>),
    Symbol {
        name: String,
    },
    LoopVar {
        name: String,
    },
}

pub fn lower_program(program: &Program) -> GraphModule {
    let mut tensors = Vec::new();
    let mut functions = Vec::new();
    let mut module_builder = GraphBuilder::new();
    let mut body_statements = Vec::new();

    for item in &program.items {
        match item {
            TopLevelDecl::TensorDecl(tensor) => tensors.push(GraphTensorDecl {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                attrs: tensor.attrs.clone(),
            }),
            TopLevelDecl::FuncDecl(func) => functions.push(lower_function(func)),
            TopLevelDecl::Stmt(stmt) => body_statements.push(module_builder.lower_stmt(stmt)),
        }
    }

    GraphModule {
        tensors,
        functions,
        body: GraphBlock {
            statements: body_statements,
        },
        nodes: module_builder.into_nodes(),
    }
}

fn lower_function(func: &FuncDecl) -> GraphFunction {
    let mut builder = GraphBuilder::new();
    let mut params = Vec::new();

    for param in &func.params {
        let node = builder.add_parameter(&param.name, param.ty.clone());
        params.push(GraphParam {
            name: param.name.clone(),
            ty: param.ty.clone(),
            node,
        });
    }

    let body = builder.lower_block(&func.body);

    GraphFunction {
        name: func.name.clone(),
        params,
        body,
        nodes: builder.into_nodes(),
        exported: func.exported,
    }
}

struct ScopeStack {
    scopes: Vec<HashMap<String, NodeId>>,
}

impl ScopeStack {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        assert!(self.scopes.len() > 1, "attempted to pop base scope");
        self.scopes.pop();
    }

    fn insert(&mut self, name: String, id: NodeId) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, id);
        }
    }

    fn get(&self, name: &str) -> Option<NodeId> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(*id);
            }
        }
        None
    }
}

struct GraphBuilder {
    nodes: Vec<Node>,
    env: ScopeStack,
    symbols: HashMap<String, NodeId>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            env: ScopeStack::new(),
            symbols: HashMap::new(),
        }
    }

    fn into_nodes(self) -> Vec<Node> {
        self.nodes
    }

    fn add_node(&mut self, kind: NodeKind) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { kind });
        id
    }

    fn add_parameter(&mut self, name: &str, ty: Option<TypeExpr>) -> NodeId {
        let id = self.add_node(NodeKind::Parameter {
            name: name.to_string(),
            ty: ty.clone(),
        });
        self.env.insert(name.to_string(), id);
        id
    }

    fn lower_block(&mut self, block: &Block) -> GraphBlock {
        self.env.push_scope();
        let result = self.lower_block_in_scope(block);
        self.env.pop_scope();
        result
    }

    fn lower_block_in_scope(&mut self, block: &Block) -> GraphBlock {
        let mut statements = Vec::new();
        for stmt in &block.stmts {
            statements.push(self.lower_stmt(stmt));
        }
        GraphBlock { statements }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> GraphStmt {
        match stmt {
            Stmt::Let { name, value } => {
                let value_id = self.lower_expr(value);
                self.env.insert(name.clone(), value_id);
                GraphStmt::Let {
                    name: name.clone(),
                    value: value_id,
                }
            }
            Stmt::Assign { name, value } => {
                let value_id = self.lower_expr(value);
                self.env.insert(name.clone(), value_id);
                GraphStmt::Assign {
                    name: name.clone(),
                    value: value_id,
                }
            }
            Stmt::Expr(expr) => GraphStmt::Expr(self.lower_expr(expr)),
            Stmt::For(for_stmt) => GraphStmt::For(self.lower_for_stmt(for_stmt)),
            Stmt::Return(expr) => GraphStmt::Return(self.lower_expr(expr)),
        }
    }

    fn lower_for_stmt(&mut self, for_stmt: &ForStmt) -> GraphForStmt {
        let iter = match &for_stmt.head.iter {
            ForIter::RangeCall(expr) => GraphForIter::RangeCall(self.lower_expr(expr)),
            ForIter::IntRange { start, end } => GraphForIter::IntRange {
                start: *start,
                end: *end,
            },
            ForIter::TupleBinding(name) => GraphForIter::TupleBinding(name.clone()),
        };

        let binding_node = self.add_node(NodeKind::LoopVar {
            name: for_stmt.head.binding.clone(),
        });

        self.env.push_scope();
        self.env.insert(for_stmt.head.binding.clone(), binding_node);
        let body = self.lower_block_in_scope(&for_stmt.body);
        self.env.pop_scope();

        GraphForStmt {
            head: GraphForHead {
                binding: for_stmt.head.binding.clone(),
                iter,
                binding_node,
            },
            body,
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> NodeId {
        match expr {
            Expr::Literal(lit) => self.add_node(NodeKind::Literal(lit.clone())),
            Expr::Ident(name) => self.lower_ident(name),
            Expr::Binary { op, left, right } => {
                let left_id = self.lower_expr(left);
                let right_id = self.lower_expr(right);
                self.add_node(NodeKind::Binary {
                    op: *op,
                    left: left_id,
                    right: right_id,
                })
            }
            Expr::Unary { op, expr } => {
                let expr_id = self.lower_expr(expr);
                self.add_node(NodeKind::Unary {
                    op: *op,
                    expr: expr_id,
                })
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_id = self.lower_expr(cond);
                let then_id = self.lower_expr(then_branch);
                let else_id = self.lower_expr(else_branch);
                self.add_node(NodeKind::If {
                    cond: cond_id,
                    then_branch: then_id,
                    else_branch: else_id,
                })
            }
            Expr::Match {
                scrutinee,
                cases,
                default,
            } => {
                let scrutinee_id = self.lower_expr(scrutinee);
                let cases = cases
                    .iter()
                    .map(|case| GraphMatchCase {
                        pattern: case.pattern.clone(),
                        value: self.lower_expr(&case.body),
                    })
                    .collect();
                let default_id = self.lower_expr(default);
                self.add_node(NodeKind::Match {
                    scrutinee: scrutinee_id,
                    cases,
                    default: default_id,
                })
            }
            Expr::Call(call) => self.lower_call_expr(call),
            Expr::Field(field) => {
                let target_id = self.lower_expr(&field.target);
                self.add_node(NodeKind::Field {
                    target: target_id,
                    field: field.field.clone(),
                })
            }
            Expr::MethodCall(method) => {
                let target_id = self.lower_expr(&method.target);
                let args = self.lower_call_args(&method.args);
                self.add_node(NodeKind::MethodCall {
                    target: target_id,
                    method: method.method.clone(),
                    args,
                })
            }
            Expr::TensorCtor(tensor_ctor) => match tensor_ctor {
                TensorCtor::Shape(dims) => self.add_node(NodeKind::TensorCtorShape(dims.clone())),
                TensorCtor::Value(value_expr) => {
                    let value_id = self.lower_expr(value_expr);
                    self.add_node(NodeKind::TensorCtorValue(value_id))
                }
            },
            Expr::Array(items) => {
                let values = items.iter().map(|item| self.lower_expr(item)).collect();
                self.add_node(NodeKind::Array(values))
            }
            Expr::Grouping(expr) => self.lower_expr(expr),
        }
    }

    fn lower_ident(&mut self, name: &str) -> NodeId {
        if let Some(id) = self.env.get(name) {
            return id;
        }
        if let Some(id) = self.symbols.get(name) {
            return *id;
        }

        let id = self.add_node(NodeKind::Symbol {
            name: name.to_string(),
        });
        self.symbols.insert(name.to_string(), id);
        id
    }

    fn lower_call_expr(&mut self, call: &CallExpr) -> NodeId {
        let callee = self.lower_expr(&call.callee);
        let args = self.lower_call_args(&call.args);
        self.add_node(NodeKind::Call { callee, args })
    }

    fn lower_call_args(&mut self, args: &[CallArg]) -> Vec<GraphCallArg> {
        args.iter()
            .map(|arg| match arg {
                CallArg::Positional(expr) => GraphCallArg::Positional(self.lower_expr(expr)),
                CallArg::Keyword { name, value } => GraphCallArg::Keyword {
                    name: name.clone(),
                    value: self.lower_expr(value),
                },
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        BinaryOp, Block, Expr, ForHead, ForIter, ForStmt, FuncDecl, Literal, Param, Program, Stmt,
        TopLevelDecl,
    };

    fn lower_single_function(func: FuncDecl) -> GraphFunction {
        let program = Program {
            items: vec![TopLevelDecl::FuncDecl(func)],
        };
        let mut module = lower_program(&program);
        assert_eq!(module.functions.len(), 1);
        module.functions.pop().unwrap()
    }

    fn ident(name: &str) -> Expr {
        Expr::Ident(name.to_string())
    }

    fn literal_int(value: i64) -> Expr {
        Expr::Literal(Literal::Int(value))
    }

    #[test]
    fn lowers_function_parameter_return() {
        let func = FuncDecl {
            exported: false,
            name: "id".to_string(),
            params: vec![Param {
                name: "x".to_string(),
                ty: None,
            }],
            body: Block {
                stmts: vec![Stmt::Return(ident("x"))],
            },
        };

        let graph_fn = lower_single_function(func);

        assert_eq!(graph_fn.name, "id");
        assert_eq!(graph_fn.params.len(), 1);
        let param = &graph_fn.params[0];
        assert_eq!(param.name, "x");
        assert_eq!(param.node, NodeId(0));
        assert!(graph_fn.nodes.len() >= 1);

        match &graph_fn.nodes[0].kind {
            NodeKind::Parameter { name, ty } => {
                assert_eq!(name, "x");
                assert!(ty.is_none());
            }
            other => panic!("expected parameter node, got {:?}", other),
        }

        match &graph_fn.body.statements[..] {
            [GraphStmt::Return(ret)] => assert_eq!(*ret, NodeId(0)),
            other => panic!("unexpected statements: {:?}", other),
        }
    }

    #[test]
    fn preserves_export_flag_on_functions() {
        let func = FuncDecl {
            exported: true,
            name: "api".to_string(),
            params: vec![],
            body: Block {
                stmts: vec![Stmt::Return(literal_int(0))],
            },
        };

        let graph_fn = lower_single_function(func);
        assert!(graph_fn.exported);
    }

    #[test]
    fn lowers_let_and_binary_expression() {
        let func = FuncDecl {
            exported: false,
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

        let graph_fn = lower_single_function(func);

        assert_eq!(graph_fn.nodes.len(), 3);

        match &graph_fn.nodes[1].kind {
            NodeKind::Literal(Literal::Int(value)) => assert_eq!(*value, 1),
            other => panic!("expected literal node, got {:?}", other),
        }

        match &graph_fn.nodes[2].kind {
            NodeKind::Binary { op, left, right } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*left, NodeId(0));
                assert_eq!(*right, NodeId(1));
            }
            other => panic!("expected binary node, got {:?}", other),
        }

        match &graph_fn.body.statements[..] {
            [GraphStmt::Let { name, value }, GraphStmt::Return(ret)] => {
                assert_eq!(name, "y");
                assert_eq!(*value, NodeId(2));
                assert_eq!(*ret, NodeId(2));
            }
            other => panic!("unexpected statements: {:?}", other),
        }
    }

    #[test]
    fn lowers_for_loop_with_binding_scope() {
        let func = FuncDecl {
            exported: false,
            name: "loop_fn".to_string(),
            params: vec![],
            body: Block {
                stmts: vec![
                    Stmt::For(ForStmt {
                        head: ForHead {
                            binding: "i".to_string(),
                            iter: ForIter::IntRange { start: 0, end: 3 },
                        },
                        body: Block {
                            stmts: vec![Stmt::Expr(ident("i"))],
                        },
                    }),
                    Stmt::Return(literal_int(0)),
                ],
            },
        };

        let graph_fn = lower_single_function(func);

        assert_eq!(graph_fn.nodes.len(), 2);

        match &graph_fn.nodes[0].kind {
            NodeKind::LoopVar { name } => assert_eq!(name, "i"),
            other => panic!("expected loop var node, got {:?}", other),
        }

        match &graph_fn.nodes[1].kind {
            NodeKind::Literal(Literal::Int(value)) => assert_eq!(*value, 0),
            other => panic!("expected literal node, got {:?}", other),
        }

        match &graph_fn.body.statements[..] {
            [GraphStmt::For(for_stmt), GraphStmt::Return(ret)] => {
                assert_eq!(for_stmt.head.binding, "i");
                assert_eq!(for_stmt.head.binding_node, NodeId(0));
                match &for_stmt.head.iter {
                    GraphForIter::IntRange { start, end } => {
                        assert_eq!((*start, *end), (0, 3));
                    }
                    other => panic!("unexpected iterator: {:?}", other),
                }

                match &for_stmt.body.statements[..] {
                    [GraphStmt::Expr(node)] => assert_eq!(*node, NodeId(0)),
                    other => panic!("unexpected loop body: {:?}", other),
                }

                assert_eq!(*ret, NodeId(1));
            }
            other => panic!("unexpected statements: {:?}", other),
        }
    }

    #[test]
    fn lowers_module_level_statements() {
        let program = Program {
            items: vec![
                TopLevelDecl::Stmt(Stmt::Let {
                    name: "g".to_string(),
                    value: literal_int(42),
                }),
                TopLevelDecl::Stmt(Stmt::Expr(ident("g"))),
            ],
        };

        let module = lower_program(&program);

        assert!(module.functions.is_empty());
        assert_eq!(module.body.statements.len(), 2);
        assert_eq!(module.nodes.len(), 1);

        match &module.body.statements[0] {
            GraphStmt::Let { name, value } => {
                assert_eq!(name, "g");
                assert_eq!(*value, NodeId(0));
            }
            other => panic!("unexpected first statement: {:?}", other),
        }

        match &module.body.statements[1] {
            GraphStmt::Expr(node) => assert_eq!(*node, NodeId(0)),
            other => panic!("unexpected second statement: {:?}", other),
        }

        match &module.nodes[0].kind {
            NodeKind::Literal(Literal::Int(value)) => assert_eq!(*value, 42),
            other => panic!("expected literal node, got {:?}", other),
        }
    }
}
