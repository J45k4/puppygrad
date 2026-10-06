use super::pop::{Error, Result};

#[derive(Clone, Debug)]
pub enum Expr {
    Name(String),
    Number(String),
    Text(String, bool),
    List(Vec<Expr>),
    Call(String, Vec<Expr>),
    Unary(char, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Name(String),
    Number(String),
    Text(String, bool),
    Symbol(String),
    End,
}

pub fn parse(source: &str) -> Result<Expr> {
    let tokens = lex(source)?;
    let mut parser = Parser {
        tokens,
        cursor: 0,
        depth: 0,
    };
    let result = parser.expr(0)?;
    if parser.peek() != &Token::End {
        return Err(Error("unexpected trailing expression tokens".into()));
    }
    Ok(result)
}

fn lex(source: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    let mut out = vec![];
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let formatted = c == 'f' && chars.get(i + 1).is_some_and(|q| matches!(q, '\'' | '"'));
        if matches!(c, '\'' | '"') || formatted {
            if formatted {
                i += 1;
            }
            let quote = chars[i];
            i += 1;
            let mut text = String::new();
            while i < chars.len() && chars[i] != quote {
                if chars[i] == '\\' {
                    i += 1;
                    let escaped = chars
                        .get(i)
                        .ok_or_else(|| Error("unterminated string escape".into()))?;
                    text.push(match escaped {
                        'n' => '\n',
                        't' => '\t',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        _ => return Err(Error("unsupported string escape".into())),
                    });
                } else {
                    text.push(chars[i]);
                }
                i += 1;
            }
            if i == chars.len() {
                return Err(Error("unterminated string".into()));
            }
            i += 1;
            out.push(Token::Text(text, formatted));
        } else if c.is_ascii_digit()
            || c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit)
        {
            let start = i;
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_digit()
                    || chars[i] == '.'
                    || matches!(chars[i], 'e' | 'E')
                    || matches!(chars[i], '+' | '-') && matches!(chars[i - 1], 'e' | 'E'))
            {
                i += 1;
            }
            out.push(Token::Number(chars[start..i].iter().collect()));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || matches!(chars[i], '_' | '.'))
            {
                i += 1;
            }
            out.push(Token::Name(chars[start..i].iter().collect()));
        } else if "()+-*/[],<>=".contains(c) {
            let mut op = c.to_string();
            i += 1;
            if chars.get(i).is_some_and(|&next| {
                c == '/' && next == '/' || matches!(c, '<' | '>' | '=') && next == '='
            }) {
                op.push(chars[i]);
                i += 1;
            }
            out.push(Token::Symbol(op));
        } else {
            return Err(Error(format!("unexpected expression character {c:?}")));
        }
    }
    out.push(Token::End);
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
    depth: usize,
}
impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }
    fn take(&mut self) -> Token {
        let token = self.tokens[self.cursor].clone();
        if token != Token::End {
            self.cursor += 1;
        }
        token
    }
    fn accept(&mut self, s: &str) -> bool {
        if self.peek() == &Token::Symbol(s.into()) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, s: &str) -> Result<()> {
        if self.accept(s) {
            Ok(())
        } else {
            Err(Error(format!("expected {s:?}")))
        }
    }
    fn list(&mut self, end: &str) -> Result<Vec<Expr>> {
        let mut values = vec![];
        if self.accept(end) {
            return Ok(values);
        }
        loop {
            values.push(self.expr(0)?);
            if self.accept(end) {
                break;
            }
            if end == "]"
                && (self.peek() == &Token::End || self.peek() == &Token::Symbol(")".into()))
            {
                return Err(Error("unclosed list".into()));
            }
            self.expect(",")?;
            if self.accept(end) {
                break;
            }
        }
        Ok(values)
    }
    fn expr(&mut self, min: u8) -> Result<Expr> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(Error("expression nesting exceeds 128".into()));
        }
        let mut lhs = match self.take() {
            Token::Number(n) => Expr::Number(n),
            Token::Text(s, f) => Expr::Text(s, f),
            Token::Name(n) => {
                if self.accept("(") {
                    Expr::Call(n, self.list(")")?)
                } else {
                    Expr::Name(n)
                }
            }
            Token::Symbol(s) if s == "[" => Expr::List(self.list("]")?),
            Token::Symbol(s) if s == "(" => {
                let e = self.expr(0)?;
                self.expect(")")?;
                e
            }
            Token::Symbol(s) if s == "-" || s == "+" => {
                Expr::Unary(s.chars().next().unwrap(), Box::new(self.expr(4)?))
            }
            _ => return Err(Error("expected an expression".into())),
        };
        loop {
            let Token::Symbol(op) = self.peek() else {
                break;
            };
            let precedence = match op.as_str() {
                "<" | ">" | "==" => 1,
                "+" | "-" => 2,
                "*" | "/" | "//" => 3,
                _ => break,
            };
            if precedence < min {
                break;
            }
            let op = op.clone();
            self.take();
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(self.expr(precedence + 1)?));
        }
        self.depth -= 1;
        Ok(lhs)
    }
}
