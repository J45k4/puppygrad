use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<TopLevelDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevelDecl {
    TensorDecl(TensorDecl),
    FuncDecl(FuncDecl),
    Stmt(Stmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorDecl {
    pub name: String,
    pub shape: Vec<Dim>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    pub name: String,
    pub value: Literal,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Int(i64),
    Ident(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Option<TypeExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    Tensor {
        dtype: Option<DType>,
        shape: Option<Vec<Dim>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let { name: String, value: Expr },
    Assign { name: String, value: Expr },
    Expr(Expr),
    For(ForStmt),
    Return(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt {
    pub head: ForHead,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForHead {
    pub binding: String,
    pub iter: ForIter,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForIter {
    RangeCall(Expr),
    IntRange { start: i64, end: i64 },
    TupleBinding(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Ident(String),
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    Match {
        scrutinee: Box<Expr>,
        cases: Vec<MatchCase>,
        default: Box<Expr>,
    },
    Call(CallExpr),
    Field(FieldExpr),
    MethodCall(MethodCallExpr),
    TensorCtor(TensorCtor),
    Array(Vec<Expr>),
    Grouping(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorCtor {
    Shape(Vec<Dim>),
    Value(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Ident(String),
    Call { callee: String, args: Vec<Pattern> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallExpr {
    pub callee: Box<Expr>,
    pub args: Vec<CallArg>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallArg {
    Positional(Expr),
    Keyword { name: String, value: Expr },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldExpr {
    pub target: Box<Expr>,
    pub field: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MethodCallExpr {
    pub target: Box<Expr>,
    pub method: String,
    pub args: Vec<CallArg>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at line {}, column {}",
            self.message, self.span.line, self.span.column
        )
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokenKind,
    lexeme: String,
    literal: Option<LiteralValue>,
    span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LiteralValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Ident,
    Int,
    Float,
    String,
    Bool,
    Keyword(Keyword),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    Dot,
    Equal,
    EqEq,
    Bang,
    BangEq,
    Plus,
    Minus,
    Star,
    Slash,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    AndAnd,
    OrOr,
    RangeDots,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keyword {
    Tensor,
    Fn,
    Let,
    For,
    In,
    Range,
    Return,
    If,
    Else,
    Match,
    Case,
    DType(DType),
}

pub fn parse(input: String) -> Result<Program, ParseError> {
    let mut lexer = Lexer::new(&input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

struct Lexer<'a> {
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    input: &'a str,
    line: usize,
    column: usize,
    last_line_start: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer {
            chars: input.char_indices().peekable(),
            input,
            line: 1,
            column: 1,
            last_line_start: 0,
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        while let Some(&(idx, ch)) = self.chars.peek() {
            if ch.is_whitespace() {
                self.consume_whitespace();
                continue;
            }

            if ch == '/' {
                if let Some((_, next)) = self.peek_next() {
                    if next == '/' {
                        self.consume_line_comment();
                        continue;
                    }
                }
            }

            let token = if ch.is_ascii_alphabetic() || ch == '_' {
                self.lex_identifier_or_keyword()?
            } else if ch.is_ascii_digit() {
                self.lex_number(false)?
            } else {
                match ch {
                    '"' => self.lex_string()?,
                    '(' => self.simple_token(TokenKind::LParen),
                    ')' => self.simple_token(TokenKind::RParen),
                    '{' => self.simple_token(TokenKind::LBrace),
                    '}' => self.simple_token(TokenKind::RBrace),
                    '[' => self.simple_token(TokenKind::LBracket),
                    ']' => self.simple_token(TokenKind::RBracket),
                    ',' => self.simple_token(TokenKind::Comma),
                    ';' => self.simple_token(TokenKind::Semi),
                    ':' => self.simple_token(TokenKind::Colon),
                    '.' => {
                        if let Some((_, next)) = self.peek_next() {
                            if next == '.' {
                                self.lex_range_dots()
                            } else if next.is_ascii_digit() {
                                self.lex_number(true)?
                            } else {
                                self.simple_token(TokenKind::Dot)
                            }
                        } else {
                            self.simple_token(TokenKind::Dot)
                        }
                    }
                    '=' => {
                        if self.peek_is('=', 1) {
                            self.consume_char(); // first '='
                            self.consume_char(); // second '='
                            self.make_token(idx, 2, TokenKind::EqEq, None)
                        } else {
                            self.simple_token(TokenKind::Equal)
                        }
                    }
                    '!' => {
                        if self.peek_is('=', 1) {
                            self.consume_char();
                            self.consume_char();
                            self.make_token(idx, 2, TokenKind::BangEq, None)
                        } else {
                            self.simple_token(TokenKind::Bang)
                        }
                    }
                    '+' => self.simple_token(TokenKind::Plus),
                    '-' => self.simple_token(TokenKind::Minus),
                    '*' => self.simple_token(TokenKind::Star),
                    '/' => self.simple_token(TokenKind::Slash),
                    '<' => {
                        if self.peek_is('=', 1) {
                            self.consume_char();
                            self.consume_char();
                            self.make_token(idx, 2, TokenKind::LessEq, None)
                        } else {
                            self.simple_token(TokenKind::Less)
                        }
                    }
                    '>' => {
                        if self.peek_is('=', 1) {
                            self.consume_char();
                            self.consume_char();
                            self.make_token(idx, 2, TokenKind::GreaterEq, None)
                        } else {
                            self.simple_token(TokenKind::Greater)
                        }
                    }
                    '&' => {
                        if self.peek_is('&', 1) {
                            self.consume_char();
                            self.consume_char();
                            self.make_token(idx, 2, TokenKind::AndAnd, None)
                        } else {
                            return Err(self.error(idx, "unexpected '&'"));
                        }
                    }
                    '|' => {
                        if self.peek_is('|', 1) {
                            self.consume_char();
                            self.consume_char();
                            self.make_token(idx, 2, TokenKind::OrOr, None)
                        } else {
                            return Err(self.error(idx, "unexpected '|'"));
                        }
                    }
                    _ => {
                        return Err(self.error(idx, &format!("unexpected character '{}'", ch)));
                    }
                }
            };
            tokens.push(token);
        }

        tokens.push(Token {
            kind: TokenKind::Eof,
            lexeme: String::new(),
            literal: None,
            span: Span {
                start: self.input.len(),
                end: self.input.len(),
                line: self.line,
                column: self.column,
            },
        });

        Ok(tokens)
    }

    fn consume_whitespace(&mut self) {
        while let Some(&(idx, ch)) = self.chars.peek() {
            if ch.is_whitespace() {
                self.consume_char();
                if ch == '\n' {
                    self.line += 1;
                    self.last_line_start = idx + ch.len_utf8();
                    self.column = 1;
                } else {
                    self.column += 1;
                }
            } else {
                break;
            }
        }
    }

    fn consume_line_comment(&mut self) {
        while let Some(&(_, ch)) = self.chars.peek() {
            self.consume_char();
            if ch == '\n' {
                self.line += 1;
                self.last_line_start = self.current_index();
                self.column = 1;
                break;
            } else {
                self.column += 1;
            }
        }
    }

    fn lex_identifier_or_keyword(&mut self) -> Result<Token, ParseError> {
        let start = self.current_index();
        let start_col = self.column;

        let mut end = start;
        let mut ident = String::new();

        while let Some(&(idx, ch)) = self.chars.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ident.push(ch);
                end = idx + ch.len_utf8();
                self.consume_char();
            } else {
                break;
            }
        }

        let (kind, literal) = match ident.as_str() {
            "Tensor" => (TokenKind::Keyword(Keyword::Tensor), None),
            "fn" => (TokenKind::Keyword(Keyword::Fn), None),
            "let" => (TokenKind::Keyword(Keyword::Let), None),
            "for" => (TokenKind::Keyword(Keyword::For), None),
            "in" => (TokenKind::Keyword(Keyword::In), None),
            "range" => (TokenKind::Keyword(Keyword::Range), None),
            "return" => (TokenKind::Keyword(Keyword::Return), None),
            "if" => (TokenKind::Keyword(Keyword::If), None),
            "else" => (TokenKind::Keyword(Keyword::Else), None),
            "match" => (TokenKind::Keyword(Keyword::Match), None),
            "case" => (TokenKind::Keyword(Keyword::Case), None),
            "f16" => (TokenKind::Keyword(Keyword::DType(DType::F16)), None),
            "bf16" => (TokenKind::Keyword(Keyword::DType(DType::BF16)), None),
            "f32" => (TokenKind::Keyword(Keyword::DType(DType::F32)), None),
            "f64" => (TokenKind::Keyword(Keyword::DType(DType::F64)), None),
            "i32" => (TokenKind::Keyword(Keyword::DType(DType::I32)), None),
            "i64" => (TokenKind::Keyword(Keyword::DType(DType::I64)), None),
            "bool" => (TokenKind::Keyword(Keyword::DType(DType::Bool)), None),
            "true" => (TokenKind::Bool, Some(LiteralValue::Bool(true))),
            "false" => (TokenKind::Bool, Some(LiteralValue::Bool(false))),
            _ => (TokenKind::Ident, None),
        };

        Ok(Token {
            kind,
            lexeme: ident,
            literal,
            span: Span {
                start,
                end,
                line: self.line,
                column: start_col,
            },
        })
    }

    fn lex_number(&mut self, started_with_dot: bool) -> Result<Token, ParseError> {
        let start = if started_with_dot {
            self.current_index()
        } else {
            self.current_index()
        };
        let start_col = self.column;

        let mut num = String::new();
        if started_with_dot {
            num.push('.');
            self.consume_char();
        }

        while let Some(&(_, ch)) = self.chars.peek() {
            if ch.is_ascii_digit() {
                num.push(ch);
                self.consume_char();
            } else {
                break;
            }
        }

        let mut is_float = started_with_dot;

        if self.peek_char() == Some('.')
            && self.peek_next_char().map_or(false, |c| c.is_ascii_digit())
        {
            is_float = true;
            num.push('.');
            self.consume_char();
            while let Some(&(_, ch)) = self.chars.peek() {
                if ch.is_ascii_digit() {
                    num.push(ch);
                    self.consume_char();
                } else {
                    break;
                }
            }
        }

        if let Some(&(_, ch)) = self.chars.peek() {
            if ch == 'e' || ch == 'E' {
                is_float = true;
                num.push(ch);
                self.consume_char();
                if let Some(&(_, sign)) = self.chars.peek() {
                    if sign == '+' || sign == '-' {
                        num.push(sign);
                        self.consume_char();
                    }
                }
                let mut has_digit = false;
                while let Some(&(_, ch)) = self.chars.peek() {
                    if ch.is_ascii_digit() {
                        num.push(ch);
                        self.consume_char();
                        has_digit = true;
                    } else {
                        break;
                    }
                }
                if !has_digit {
                    return Err(self.error(start, "invalid float exponent"));
                }
            }
        }

        let end = self.current_index();
        if is_float {
            let value = num
                .parse::<f64>()
                .map_err(|_| self.error(start, "invalid float literal"))?;
            Ok(Token {
                kind: TokenKind::Float,
                lexeme: num,
                literal: Some(LiteralValue::Float(value)),
                span: Span {
                    start,
                    end,
                    line: self.line,
                    column: start_col,
                },
            })
        } else {
            let value = num
                .parse::<i64>()
                .map_err(|_| self.error(start, "invalid integer literal"))?;
            Ok(Token {
                kind: TokenKind::Int,
                lexeme: num,
                literal: Some(LiteralValue::Int(value)),
                span: Span {
                    start,
                    end,
                    line: self.line,
                    column: start_col,
                },
            })
        }
    }

    fn lex_range_dots(&mut self) -> Token {
        let start_idx = self.current_index();
        let start_col = self.column;
        self.consume_char(); // first dot
        self.consume_char(); // second dot
        Token {
            kind: TokenKind::RangeDots,
            lexeme: "..".to_string(),
            literal: None,
            span: Span {
                start: start_idx,
                end: self.current_index(),
                line: self.line,
                column: start_col,
            },
        }
    }

    fn lex_string(&mut self) -> Result<Token, ParseError> {
        let start_idx = self.current_index();
        let start_col = self.column;
        self.consume_char(); // opening quote
        let mut value = String::new();
        while let Some(&(_, ch)) = self.chars.peek() {
            self.consume_char();
            match ch {
                '"' => {
                    let end = self.current_index();
                    return Ok(Token {
                        kind: TokenKind::String,
                        lexeme: value.clone(),
                        literal: Some(LiteralValue::Str),
                        span: Span {
                            start: start_idx,
                            end,
                            line: self.line,
                            column: start_col,
                        },
                    });
                }
                '\\' => {
                    if let Some(&(_, esc)) = self.chars.peek() {
                        self.consume_char();
                        match esc {
                            '"' => value.push('"'),
                            '\\' => value.push('\\'),
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            'r' => value.push('\r'),
                            other => {
                                return Err(self.error(
                                    start_idx,
                                    &format!("invalid escape sequence '\\{}'", other),
                                ));
                            }
                        }
                    } else {
                        return Err(self.error(start_idx, "unterminated escape sequence"));
                    }
                }
                '\n' => {
                    return Err(self.error(start_idx, "newline in string literal"));
                }
                other => value.push(other),
            }
        }
        Err(self.error(start_idx, "unterminated string literal"))
    }

    fn simple_token(&mut self, kind: TokenKind) -> Token {
        let idx = self.current_index();
        let column = self.column;
        self.consume_char();
        Token {
            kind,
            lexeme: self.input[idx..self.current_index()].to_string(),
            literal: None,
            span: Span {
                start: idx,
                end: self.current_index(),
                line: self.line,
                column,
            },
        }
    }

    fn make_token(
        &self,
        start: usize,
        len: usize,
        kind: TokenKind,
        literal: Option<LiteralValue>,
    ) -> Token {
        Token {
            kind,
            lexeme: self.input[start..start + len].to_string(),
            literal,
            span: Span {
                start,
                end: start + len,
                line: self.line,
                column: self.column,
            },
        }
    }

    fn consume_char(&mut self) {
        if let Some((idx, ch)) = self.chars.next() {
            if ch == '\n' {
                self.line += 1;
                self.last_line_start = idx + ch.len_utf8();
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    fn peek_next(&mut self) -> Option<(usize, char)> {
        let mut clone = self.chars.clone();
        clone.next();
        clone.peek().copied()
    }

    fn peek_is(&mut self, expected: char, offset: usize) -> bool {
        let mut clone = self.chars.clone();
        for _ in 0..offset {
            if clone.next().is_none() {
                return false;
            }
        }
        clone.peek().map(|(_, ch)| *ch == expected).unwrap_or(false)
    }

    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, ch)| *ch)
    }

    fn peek_next_char(&mut self) -> Option<char> {
        self.peek_next().map(|(_, ch)| ch)
    }

    fn current_index(&mut self) -> usize {
        self.chars
            .peek()
            .map(|(idx, _)| *idx)
            .unwrap_or(self.input.len())
    }

    fn error(&self, index: usize, message: &str) -> ParseError {
        let line = self.line;
        let column = self.column;
        ParseError {
            message: message.to_string(),
            span: Span {
                start: index,
                end: index,
                line,
                column,
            },
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        while !self.is_at_end() {
            if self.check_kind(TokenKind::Eof) {
                break;
            }
            items.push(self.parse_top_level_decl()?);
        }
        Ok(Program { items })
    }

    fn parse_top_level_decl(&mut self) -> Result<TopLevelDecl, ParseError> {
        if self.check_keyword(Keyword::Fn) {
            Ok(TopLevelDecl::FuncDecl(self.parse_func_decl()?))
        } else if self.check_keyword(Keyword::Tensor)
            && self.lookahead_is_ident()
            && self.lookahead_kind(2, TokenKind::LParen)
        {
            Ok(TopLevelDecl::TensorDecl(self.parse_tensor_decl()?))
        } else {
            Ok(TopLevelDecl::Stmt(self.parse_stmt()?))
        }
    }

    fn parse_tensor_decl(&mut self) -> Result<TensorDecl, ParseError> {
        self.consume_keyword(Keyword::Tensor, "expected 'Tensor'")?;
        let name = self.consume_ident("expected tensor name")?;
        self.consume(TokenKind::LParen, "expected '(' after tensor name")?;
        let shape = self.parse_shape_list()?;
        self.consume(TokenKind::RParen, "expected ')' after tensor shape")?;
        let mut attrs = Vec::new();
        if self.match_kind(TokenKind::Comma) {
            attrs = self.parse_attr_list()?;
        }
        Ok(TensorDecl { name, shape, attrs })
    }

    fn parse_attr_list(&mut self) -> Result<Vec<Attr>, ParseError> {
        let mut attrs = Vec::new();
        loop {
            let name = self.consume_ident("expected attribute name")?;
            self.consume(TokenKind::Equal, "expected '=' in attribute")?;
            let value = self.parse_literal()?;
            attrs.push(Attr { name, value });
            if !self.match_kind(TokenKind::Comma) {
                break;
            }
        }
        Ok(attrs)
    }

    fn parse_func_decl(&mut self) -> Result<FuncDecl, ParseError> {
        self.consume_keyword(Keyword::Fn, "expected 'fn'")?;
        let name = self.consume_ident("expected function name")?;
        self.consume(TokenKind::LParen, "expected '(' after function name")?;
        let params = if self.check_kind(TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_param_list()?
        };
        self.consume(TokenKind::RParen, "expected ')' after parameters")?;
        let body = self.parse_block()?;
        Ok(FuncDecl { name, params, body })
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        loop {
            params.push(self.parse_param()?);
            if !self.match_kind(TokenKind::Comma) {
                break;
            }
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let name = self.consume_ident("expected parameter name")?;
        let ty = if self.match_kind(TokenKind::Colon) {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        Ok(Param { name, ty })
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        self.consume_keyword(Keyword::Tensor, "expected 'Tensor' in type")?;
        let mut dtype = None;
        let mut shape = None;
        if self.match_kind(TokenKind::LBracket) {
            if let Some(dtype_val) = self.try_parse_dtype()? {
                dtype = Some(dtype_val);
                if self.match_kind(TokenKind::Comma) {
                    shape = Some(self.parse_shape()?);
                }
            } else if !self.check_kind(TokenKind::RBracket) {
                shape = Some(self.parse_shape()?);
            }
            self.consume(TokenKind::RBracket, "expected ']' after tensor type")?;
        }
        Ok(TypeExpr::Tensor { dtype, shape })
    }

    fn try_parse_dtype(&mut self) -> Result<Option<DType>, ParseError> {
        if let Some(keyword) = self.peek_keyword() {
            if let Keyword::DType(dtype) = keyword {
                self.advance();
                return Ok(Some(dtype));
            }
        }
        Ok(None)
    }

    fn parse_shape(&mut self) -> Result<Vec<Dim>, ParseError> {
        let mut dims = Vec::new();
        loop {
            dims.push(self.parse_dim()?);
            if !self.match_kind(TokenKind::Comma) {
                break;
            }
        }
        Ok(dims)
    }

    fn parse_shape_list(&mut self) -> Result<Vec<Dim>, ParseError> {
        if self.check_kind(TokenKind::RParen) {
            return Ok(Vec::new());
        }
        self.parse_shape()
    }

    fn parse_dim(&mut self) -> Result<Dim, ParseError> {
        if self.check_kind(TokenKind::Int) {
            let tok = self.advance().clone();
            let value = tok.literal_as_int().ok_or_else(|| ParseError {
                message: "invalid integer literal".to_string(),
                span: tok.span,
            })?;
            Ok(Dim::Int(value))
        } else {
            let name = self.consume_ident("expected dimension identifier")?;
            Ok(Dim::Ident(name))
        }
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        self.consume(TokenKind::LBrace, "expected '{' to start block")?;
        let mut stmts = Vec::new();
        while !self.check_kind(TokenKind::RBrace) && !self.check_kind(TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.consume(TokenKind::RBrace, "expected '}' to end block")?;
        Ok(Block { stmts })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        if self.match_keyword(Keyword::Let) {
            let name = self.consume_ident("expected identifier after 'let'")?;
            self.consume(TokenKind::Equal, "expected '=' in let statement")?;
            let value = self.parse_expr()?;
            self.consume(TokenKind::Semi, "expected ';' after let statement")?;
            return Ok(Stmt::Let { name, value });
        }

        if self.match_keyword(Keyword::For) {
            let head = self.parse_for_head()?;
            let body = self.parse_block()?;
            return Ok(Stmt::For(ForStmt { head, body }));
        }

        if self.match_keyword(Keyword::Return) {
            let value = self.parse_expr()?;
            self.consume(TokenKind::Semi, "expected ';' after return")?;
            return Ok(Stmt::Return(value));
        }

        if self.peek_is_ident() && self.lookahead_kind(1, TokenKind::Equal) {
            let name = self.consume_ident("expected identifier in assignment")?;
            self.consume(TokenKind::Equal, "expected '=' in assignment")?;
            let value = self.parse_expr()?;
            self.consume(TokenKind::Semi, "expected ';' after assignment")?;
            return Ok(Stmt::Assign { name, value });
        }

        let expr = self.parse_expr()?;
        self.consume(TokenKind::Semi, "expected ';' after expression statement")?;
        Ok(Stmt::Expr(expr))
    }

    fn parse_for_head(&mut self) -> Result<ForHead, ParseError> {
        let binding = self.consume_ident("expected loop binding identifier")?;
        self.consume_keyword(Keyword::In, "expected 'in' in for loop")?;
        let iter = if self.match_keyword(Keyword::Range) {
            self.consume(TokenKind::LParen, "expected '(' after range")?;
            let spec = self.parse_expr()?;
            self.consume(TokenKind::RParen, "expected ')' after range spec")?;
            ForIter::RangeCall(spec)
        } else if self.check_kind(TokenKind::Int) && self.lookahead_kind(1, TokenKind::RangeDots) {
            let start_token = self.advance().clone();
            let start = start_token.literal_as_int().ok_or_else(|| ParseError {
                message: "invalid integer literal".to_string(),
                span: start_token.span,
            })?;
            self.consume(TokenKind::RangeDots, "expected '..' in range")?;
            let end_token = self
                .consume(TokenKind::Int, "expected integer after '..'")?
                .clone();
            let end = end_token.literal_as_int().ok_or_else(|| ParseError {
                message: "invalid integer literal".to_string(),
                span: end_token.span,
            })?;
            ForIter::IntRange { start, end }
        } else {
            self.consume(TokenKind::LParen, "expected '(' after 'in'")?;
            let name = self.consume_ident("expected identifier inside parentheses")?;
            self.consume(TokenKind::RParen, "expected ')' after identifier")?;
            ForIter::TupleBinding(name)
        };
        Ok(ForHead { binding, iter })
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        if self.match_keyword(Keyword::If) {
            return self.parse_if_expr();
        }
        if self.match_keyword(Keyword::Match) {
            return self.parse_match_expr();
        }
        self.parse_logic_or()
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        self.consume(TokenKind::LParen, "expected '(' after 'if'")?;
        let cond = self.parse_expr()?;
        self.consume(TokenKind::RParen, "expected ')' after condition")?;
        let then_branch = self.parse_expr()?;
        self.consume_keyword(Keyword::Else, "expected 'else' after if branch")?;
        let else_branch = self.parse_expr()?;
        Ok(Expr::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        })
    }

    fn parse_match_expr(&mut self) -> Result<Expr, ParseError> {
        let scrutinee = self.parse_expr()?;
        self.consume(TokenKind::LBrace, "expected '{' after match expression")?;
        let mut cases = Vec::new();
        while self.match_keyword(Keyword::Case) {
            let pattern = self.parse_pattern()?;
            self.consume(TokenKind::Colon, "expected ':' after pattern")?;
            let body = self.parse_expr()?;
            cases.push(MatchCase { pattern, body });
        }
        self.consume_keyword(Keyword::Else, "expected 'else' in match")?;
        self.consume(TokenKind::Colon, "expected ':' after else")?;
        let default = self.parse_expr()?;
        self.consume(TokenKind::RBrace, "expected '}' after match expression")?;
        Ok(Expr::Match {
            scrutinee: Box::new(scrutinee),
            cases,
            default: Box::new(default),
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        if self.check_kind(TokenKind::Ident) {
            if self.peek().lexeme == "_" {
                self.advance();
                return Ok(Pattern::Wildcard);
            }
            if self.lookahead_kind(1, TokenKind::LParen) {
                let callee = self.consume_ident("expected pattern name")?;
                self.consume(TokenKind::LParen, "expected '(' after pattern name")?;
                let args = if self.check_kind(TokenKind::RParen) {
                    Vec::new()
                } else {
                    self.parse_pattern_list()?
                };
                self.consume(TokenKind::RParen, "expected ')' after pattern arguments")?;
                return Ok(Pattern::Call { callee, args });
            }
            let name = self.consume_ident("expected pattern identifier")?;
            return Ok(Pattern::Ident(name));
        }

        if self.check_kind(TokenKind::Int)
            || self.check_kind(TokenKind::Float)
            || self.check_kind(TokenKind::Bool)
            || self.check_kind(TokenKind::String)
        {
            let literal = self.parse_literal()?;
            return Ok(Pattern::Literal(literal));
        }

        Err(self.error_here("expected pattern"))
    }

    fn parse_pattern_list(&mut self) -> Result<Vec<Pattern>, ParseError> {
        let mut patterns = Vec::new();
        loop {
            patterns.push(self.parse_pattern()?);
            if !self.match_kind(TokenKind::Comma) {
                break;
            }
        }
        Ok(patterns)
    }

    fn parse_logic_or(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_logic_and()?;
        while self.match_kind(TokenKind::OrOr) {
            let right = self.parse_logic_and()?;
            expr = Expr::Binary {
                op: BinaryOp::Or,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_logic_and(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_equality()?;
        while self.match_kind(TokenKind::AndAnd) {
            let right = self.parse_equality()?;
            expr = Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_relational()?;
        loop {
            if self.match_kind(TokenKind::EqEq) {
                let right = self.parse_relational()?;
                expr = Expr::Binary {
                    op: BinaryOp::Eq,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else if self.match_kind(TokenKind::BangEq) {
                let right = self.parse_relational()?;
                expr = Expr::Binary {
                    op: BinaryOp::NotEq,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_relational(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_addition()?;
        loop {
            if self.match_kind(TokenKind::Less) {
                let right = self.parse_addition()?;
                expr = Expr::Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else if self.match_kind(TokenKind::LessEq) {
                let right = self.parse_addition()?;
                expr = Expr::Binary {
                    op: BinaryOp::LtEq,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else if self.match_kind(TokenKind::Greater) {
                let right = self.parse_addition()?;
                expr = Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else if self.match_kind(TokenKind::GreaterEq) {
                let right = self.parse_addition()?;
                expr = Expr::Binary {
                    op: BinaryOp::GtEq,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_addition(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_multiplication()?;
        loop {
            if self.match_kind(TokenKind::Plus) {
                let right = self.parse_multiplication()?;
                expr = Expr::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else if self.match_kind(TokenKind::Minus) {
                let right = self.parse_multiplication()?;
                expr = Expr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_multiplication(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary()?;
        loop {
            if self.match_kind(TokenKind::Star) {
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else if self.match_kind(TokenKind::Slash) {
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    op: BinaryOp::Div,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.match_kind(TokenKind::Minus) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(expr),
            });
        }
        if self.match_kind(TokenKind::Bang) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(expr),
            });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.match_kind(TokenKind::LParen) {
                let args = if self.check_kind(TokenKind::RParen) {
                    Vec::new()
                } else {
                    self.parse_arg_list()?
                };
                self.consume(TokenKind::RParen, "expected ')' after arguments")?;
                expr = Expr::Call(CallExpr {
                    callee: Box::new(expr),
                    args,
                });
            } else if self.match_kind(TokenKind::Dot) {
                let field = self.consume_ident("expected field name after '.'")?;
                if self.match_kind(TokenKind::LParen) {
                    let args = if self.check_kind(TokenKind::RParen) {
                        Vec::new()
                    } else {
                        self.parse_arg_list()?
                    };
                    self.consume(TokenKind::RParen, "expected ')' after arguments")?;
                    expr = Expr::MethodCall(MethodCallExpr {
                        target: Box::new(expr),
                        method: field,
                        args,
                    });
                } else {
                    expr = Expr::Field(FieldExpr {
                        target: Box::new(expr),
                        field,
                    });
                }
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        if self.match_kind(TokenKind::Int)
            || self.match_kind(TokenKind::Float)
            || self.match_kind(TokenKind::Bool)
            || self.match_kind(TokenKind::String)
        {
            let token = self.previous().clone();
            let literal = self.literal_from_token(&token)?;
            return Ok(Expr::Literal(literal));
        }

        if self.match_kind(TokenKind::Ident) {
            let name = self.previous().lexeme.clone();
            return Ok(Expr::Ident(name));
        }

        if self.match_kind(TokenKind::LParen) {
            let expr = self.parse_expr()?;
            self.consume(TokenKind::RParen, "expected ')' after expression")?;
            return Ok(Expr::Grouping(Box::new(expr)));
        }

        if self.check_keyword(Keyword::Tensor) && self.lookahead_kind(1, TokenKind::LParen) {
            self.consume_keyword(Keyword::Tensor, "expected 'Tensor'")?;
            self.consume(TokenKind::LParen, "expected '(' after Tensor")?;
            if self.check_kind(TokenKind::RParen) {
                self.consume(TokenKind::RParen, "expected ')' after tensor constructor")?;
                return Ok(Expr::TensorCtor(TensorCtor::Shape(Vec::new())));
            }

            if self.check_kind(TokenKind::LBracket) {
                let value = self.parse_expr()?;
                self.consume(TokenKind::RParen, "expected ')' after tensor constructor")?;
                return Ok(Expr::TensorCtor(TensorCtor::Value(Box::new(value))));
            }

            let shape = self.parse_shape_list()?;
            self.consume(TokenKind::RParen, "expected ')' after tensor constructor")?;
            return Ok(Expr::TensorCtor(TensorCtor::Shape(shape)));
        }

        if self.match_kind(TokenKind::LBracket) {
            let mut elements = Vec::new();
            if !self.check_kind(TokenKind::RBracket) {
                loop {
                    elements.push(self.parse_expr()?);
                    if !self.match_kind(TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.consume(TokenKind::RBracket, "expected ']' after array literal")?;
            return Ok(Expr::Array(elements));
        }

        Err(self.error_here("expected expression"))
    }

    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        if self.match_kind(TokenKind::Int)
            || self.match_kind(TokenKind::Float)
            || self.match_kind(TokenKind::Bool)
            || self.match_kind(TokenKind::String)
        {
            let token = self.previous().clone();
            self.literal_from_token(&token)
        } else {
            Err(self.error_here("expected literal"))
        }
    }

    fn literal_from_token(&self, token: &Token) -> Result<Literal, ParseError> {
        match token.kind {
            TokenKind::Int => Ok(Literal::Int(token.literal_as_int().ok_or_else(|| {
                ParseError {
                    message: "invalid integer literal".to_string(),
                    span: token.span,
                }
            })?)),
            TokenKind::Float => Ok(Literal::Float(token.literal_as_float().ok_or_else(
                || ParseError {
                    message: "invalid float literal".to_string(),
                    span: token.span,
                },
            )?)),
            TokenKind::Bool => Ok(Literal::Bool(token.literal_as_bool().ok_or_else(|| {
                ParseError {
                    message: "invalid bool literal".to_string(),
                    span: token.span,
                }
            })?)),
            TokenKind::String => Ok(Literal::String(token.lexeme.clone())),
            _ => Err(ParseError {
                message: "expected literal".to_string(),
                span: token.span,
            }),
        }
    }

    fn parse_arg_list(&mut self) -> Result<Vec<CallArg>, ParseError> {
        let mut args = Vec::new();
        loop {
            if self.peek_is_ident() && self.lookahead_kind(1, TokenKind::Equal) {
                let name = self.consume_ident("expected argument name")?;
                self.consume(TokenKind::Equal, "expected '=' in keyword argument")?;
                let value = self.parse_expr()?;
                args.push(CallArg::Keyword { name, value });
            } else {
                let expr = self.parse_expr()?;
                args.push(CallArg::Positional(expr));
            }
            if !self.match_kind(TokenKind::Comma) {
                break;
            }
        }
        Ok(args)
    }

    fn match_kind(&mut self, kind: TokenKind) -> bool {
        if self.check_kind(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn match_keyword(&mut self, keyword: Keyword) -> bool {
        if self.check_keyword(keyword) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token, ParseError> {
        if self.check_kind(kind) {
            Ok(self.advance())
        } else {
            Err(self.error_here(message))
        }
    }

    fn consume_keyword(&mut self, keyword: Keyword, message: &str) -> Result<&Token, ParseError> {
        if self.check_keyword(keyword) {
            Ok(self.advance())
        } else {
            Err(self.error_here(message))
        }
    }

    fn consume_ident(&mut self, message: &str) -> Result<String, ParseError> {
        if self.check_kind(TokenKind::Ident) {
            let token = self.advance();
            Ok(token.lexeme.clone())
        } else {
            Err(self.error_here(message))
        }
    }

    fn check_kind(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn check_keyword(&self, keyword: Keyword) -> bool {
        matches!(self.peek().kind, TokenKind::Keyword(k) if k == keyword)
    }

    fn peek_keyword(&self) -> Option<Keyword> {
        if let TokenKind::Keyword(k) = self.peek().kind {
            Some(k)
        } else {
            None
        }
    }

    fn lookahead_kind(&self, offset: usize, kind: TokenKind) -> bool {
        self.tokens
            .get(self.current + offset)
            .map(|token| token.kind == kind)
            .unwrap_or(false)
    }

    fn lookahead_is_ident(&self) -> bool {
        self.tokens
            .get(self.current + 1)
            .map(|token| token.kind == TokenKind::Ident)
            .unwrap_or(false)
    }

    fn peek_is_ident(&self) -> bool {
        self.peek().kind == TokenKind::Ident
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn error_here(&self, message: &str) -> ParseError {
        let span = self.peek().span;
        ParseError {
            message: message.to_string(),
            span,
        }
    }
}

impl Token {
    fn literal_as_int(&self) -> Option<i64> {
        match self.literal {
            Some(LiteralValue::Int(value)) => Some(value),
            _ => None,
        }
    }

    fn literal_as_float(&self) -> Option<f64> {
        match self.literal {
            Some(LiteralValue::Float(value)) => Some(value),
            _ => None,
        }
    }

    fn literal_as_bool(&self) -> Option<bool> {
        match self.literal {
            Some(LiteralValue::Bool(value)) => Some(value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_program(src: &str) -> Program {
        parse(src.to_string()).expect("parse failed")
    }

    #[test]
    fn parses_tensor_decl_with_attrs() {
        let program = parse_program(r#"Tensor weights(3, 4), dtype="f32""#);
        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelDecl::TensorDecl(decl) => {
                assert_eq!(decl.name, "weights");
                assert_eq!(decl.shape.len(), 2);
                assert_eq!(decl.attrs.len(), 1);
                assert_eq!(decl.attrs[0].name, "dtype");
                assert!(matches!(
                    &decl.attrs[0].value,
                    Literal::String(value) if value == "f32"
                ));
            }
            other => panic!("expected tensor declaration, found {:?}", other),
        }
    }

    #[test]
    fn parses_simple_function() {
        let program = parse_program(
            r#"
                fn add(a: Tensor[f32, N], b: Tensor[f32, N]) {
                    return a + b;
                }
            "#,
        );
        assert_eq!(program.items.len(), 1);
        assert!(matches!(program.items[0], TopLevelDecl::FuncDecl(_)));
    }

    #[test]
    fn parses_for_with_range_call() {
        let program = parse_program(
            r#"
                for i in range(10) {
                    return i;
                }
            "#,
        );
        assert_eq!(program.items.len(), 1);

        let for_stmt = match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::For(for_stmt)) => for_stmt,
            other => panic!("expected for-statement, found {:?}", other),
        };

        assert_eq!(for_stmt.head.binding, "i");
        match &for_stmt.head.iter {
            ForIter::RangeCall(expr) => match expr {
                Expr::Literal(Literal::Int(value)) => assert_eq!(*value, 10),
                other => panic!("expected literal range bound, found {:?}", other),
            },
            other => panic!("expected range call iterator, found {:?}", other),
        }

        assert_eq!(for_stmt.body.stmts.len(), 1);
        match &for_stmt.body.stmts[0] {
            Stmt::Return(Expr::Ident(name)) if name == "i" => {}
            other => panic!("expected return of loop binding, found {:?}", other),
        }
    }

    #[test]
    fn parses_for_with_int_range() {
        let program = parse_program(
            r#"
                for idx in 0..4 {
                    let doubled = idx + idx;
                }
            "#,
        );
        assert_eq!(program.items.len(), 1);

        let for_stmt = match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::For(for_stmt)) => for_stmt,
            other => panic!("expected for-statement, found {:?}", other),
        };

        assert_eq!(for_stmt.head.binding, "idx");
        match &for_stmt.head.iter {
            ForIter::IntRange { start, end } => assert_eq!((*start, *end), (0, 4)),
            other => panic!("expected integer range iterator, found {:?}", other),
        }

        assert_eq!(for_stmt.body.stmts.len(), 1);
        match &for_stmt.body.stmts[0] {
            Stmt::Let { name, value } => {
                assert_eq!(name, "doubled");
                match value {
                    Expr::Binary {
                        op: BinaryOp::Add,
                        left,
                        right,
                    } => {
                        assert!(matches!(left.as_ref(), Expr::Ident(id) if id == "idx"));
                        assert!(matches!(right.as_ref(), Expr::Ident(id) if id == "idx"));
                    }
                    other => panic!("expected addition expr, found {:?}", other),
                }
            }
            other => panic!("expected let statement, found {:?}", other),
        }
    }

    #[test]
    fn parses_match_expression_with_cases() {
        let program = parse_program(
            r#"
                let label = match score {
                    case 0: "zero"
                    case 1: "one"
                    else: "many"
                };
            "#,
        );
        assert_eq!(program.items.len(), 1);

        let (name, value) = match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::Let { name, value }) => (name, value),
            other => panic!("expected let statement, found {:?}", other),
        };
        assert_eq!(name, "label");

        let Expr::Match {
            scrutinee,
            cases,
            default,
        } = value
        else {
            panic!("expected match expression, found {:?}", value);
        };

        assert!(matches!(scrutinee.as_ref(), Expr::Ident(id) if id == "score"));
        assert_eq!(cases.len(), 2);

        match &cases[0] {
            MatchCase {
                pattern: Pattern::Literal(Literal::Int(0)),
                body: Expr::Literal(Literal::String(text)),
            } => assert_eq!(text, "zero"),
            other => panic!("unexpected first case {:?}", other),
        }

        match &cases[1] {
            MatchCase {
                pattern: Pattern::Literal(Literal::Int(1)),
                body: Expr::Literal(Literal::String(text)),
            } => assert_eq!(text, "one"),
            other => panic!("unexpected second case {:?}", other),
        }

        match &**default {
            Expr::Literal(Literal::String(text)) => assert_eq!(text, "many"),
            other => panic!("unexpected default {:?}", other),
        }
    }

    #[test]
    fn respects_operator_precedence() {
        let program = parse_program("let value = 1 + 2 * 3 == 7 && false || true;");
        assert_eq!(program.items.len(), 1);

        let value_expr = match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::Let { value, .. }) => value,
            other => panic!("expected let statement, found {:?}", other),
        };

        let Expr::Binary {
            op: BinaryOp::Or,
            left,
            right,
        } = value_expr
        else {
            panic!("expected top-level 'or' binary, found {:?}", value_expr);
        };
        assert!(matches!(**right, Expr::Literal(Literal::Bool(true))));

        let Expr::Binary {
            op: BinaryOp::And,
            left: and_left,
            right: and_right,
        } = &**left
        else {
            panic!("expected 'and' binary on left, found {:?}", left);
        };
        assert!(matches!(**and_right, Expr::Literal(Literal::Bool(false))));

        let Expr::Binary {
            op: BinaryOp::Eq,
            left: eq_left,
            right: eq_right,
        } = &**and_left
        else {
            panic!("expected equality expression, found {:?}", and_left);
        };
        assert!(matches!(**eq_right, Expr::Literal(Literal::Int(7))));

        let Expr::Binary {
            op: BinaryOp::Add,
            left: add_left,
            right: add_right,
        } = &**eq_left
        else {
            panic!("expected addition expression, found {:?}", eq_left);
        };
        assert!(matches!(**add_left, Expr::Literal(Literal::Int(1))));

        let Expr::Binary {
            op: BinaryOp::Mul,
            left: mul_left,
            right: mul_right,
        } = &**add_right
        else {
            panic!("expected multiplication expression, found {:?}", add_right);
        };

        assert!(matches!(mul_left.as_ref(), Expr::Literal(Literal::Int(2))));
        assert!(matches!(mul_right.as_ref(), Expr::Literal(Literal::Int(3))));
    }

    #[test]
    fn parses_method_call_with_keyword_argument() {
        let program = parse_program("let y = tensor.reshape(3, 4).sum(axis=1);");
        assert_eq!(program.items.len(), 1);

        let expr = match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::Let { value, .. }) => value,
            other => panic!("expected let statement, found {:?}", other),
        };

        let Expr::MethodCall(MethodCallExpr {
            target,
            method,
            args,
        }) = expr
        else {
            panic!("expected method call expression, found {:?}", expr);
        };
        assert_eq!(method, "sum");
        assert_eq!(args.len(), 1);
        match &args[0] {
            CallArg::Keyword { name, value } => {
                assert_eq!(name, "axis");
                assert!(matches!(value, Expr::Literal(Literal::Int(1))));
            }
            other => panic!("expected keyword arg, found {:?}", other),
        }

        let Expr::MethodCall(MethodCallExpr {
            target: inner_target,
            method: inner_method,
            args: inner_args,
        }) = &**target
        else {
            panic!("expected reshape method call, found {:?}", target);
        };

        assert_eq!(inner_method, "reshape");
        assert_eq!(inner_args.len(), 2);
        assert!(matches!(
            inner_args[0],
            CallArg::Positional(Expr::Literal(Literal::Int(3)))
        ));
        assert!(matches!(
            inner_args[1],
            CallArg::Positional(Expr::Literal(Literal::Int(4)))
        ));

        assert!(matches!(inner_target.as_ref(), Expr::Ident(id) if id == "tensor"));
    }

    #[test]
    fn parses_tensor_ctor_with_array_literal() {
        let program = parse_program("let x = Tensor([1, 2, 3, 4, 5]);");
        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::Let { value, .. }) => match value {
                Expr::TensorCtor(TensorCtor::Value(array_expr)) => match &**array_expr {
                    Expr::Array(elements) => {
                        assert_eq!(elements.len(), 5);
                    }
                    other => panic!("expected array literal, found {:?}", other),
                },
                other => panic!("expected tensor ctor with value, found {:?}", other),
            },
            other => panic!("expected let statement, found {:?}", other),
        }
    }

    #[test]
    fn parses_tensor_ctor_with_nested_array_literal() {
        let program = parse_program("let x = Tensor([[1, 2, 3], [1, 2, 3]]);");
        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelDecl::Stmt(Stmt::Let { value, .. }) => match value {
                Expr::TensorCtor(TensorCtor::Value(array_expr)) => match &**array_expr {
                    Expr::Array(rows) => {
                        assert_eq!(rows.len(), 2);
                        for row in rows {
                            match row {
                                Expr::Array(cols) => {
                                    assert_eq!(cols.len(), 3);
                                }
                                other => panic!("expected nested array, found {:?}", other),
                            }
                        }
                    }
                    other => panic!("expected array literal, found {:?}", other),
                },
                other => panic!("expected tensor ctor with value, found {:?}", other),
            },
            other => panic!("expected let statement, found {:?}", other),
        }
    }
}
