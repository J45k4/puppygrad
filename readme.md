# Puppygrad

## Grammar

```
letter        = "A"… "Z" | "a"… "z" | "_" ;
digit         = "0"… "9" ;
hexDigit      = digit | "A"…"F" | "a"…"f" ;

IDENT         = letter , { letter | digit } ;
INT           = digit , { digit } ;
FLOAT         = INT , "." , { digit } , [ exponent ] | "." , digit , { digit } , [ exponent ] ;
exponent      = ( "e" | "E" ) , [ "+" | "-" ] , INT ;
BOOL          = "true" | "false" ;
STRING        = "\"" , { anyCharExceptQuoteOrNewline | escape } , "\"" ;
escape        = "\\" , ( "\"" | "\\" | "n" | "t" | "r" ) ;

Program       = { TopLevelDecl } ;
TopLevelDecl  =
      TensorDecl
    | FuncDecl
    | Stmt
    ;
    
Type          = "Tensor" [ "[" DType [ "," Shape ] "]" ] ;
DType         = "f16" | "bf16" | "f32" | "f64" | "i32" | "i64" | "bool" ;
Shape         = Dim { "," Dim } ;
Dim           = INT | IDENT ;

TensorDecl    = "Tensor" IDENT "(" ShapeList ")" [ TensorAttrs ] ;
ShapeList     = Dim { "," Dim } ;
TensorAttrs   = "," AttrList ;
AttrList      = Attr { "," Attr } ;
Attr          = IDENT "=" AttrValue ;
AttrValue     = INT | FLOAT | BOOL | STRING ;

KVArgsOpt     = [ KVArg { "," KVArg } ] ;
KVArg         = IDENT "=" Expr ;

FuncDecl      = "fn" IDENT "(" ParamListOpt ")" Block ;
ParamListOpt  = [ Param { "," Param } ] ;
Param         = IDENT [ ":" Type ] ;

Block         = "{" { Stmt } "}" ;
Stmt          =
      LetStmt ";"
    | AssignStmt ";"
    | ExprStmt ";"
    | ForStmt
    | ReturnStmt ";"
    ;

LetStmt       = "let" IDENT "=" Expr ;
AssignStmt    = IDENT "=" Expr ;
ExprStmt      = Expr ;
ReturnStmt    = "return" Expr ;

ForStmt       = "for" ForHead Block ;
ForHead       = IDENT "in" ( "range" "(" RangeSpec ")" | INT ".." INT | "(" IDENT ")" ) ;
RangeSpec     = Expr ;

Expr          =
    | IfExpr
    | MatchExpr
    | LogicOr
    ;

IfExpr        = "if" "(" Expr ")" Expr "else" Expr ;
MatchExpr     = "match" Expr "{" { CaseClause } "else" ":" Expr "}" ;
CaseClause    = "case" Pattern ":" Expr ;

Pattern       = "_" | Literal | IDENT | CallPat ;
CallPat       = IDENT "(" PatListOpt ")" ;
PatListOpt    = [ Pattern { "," Pattern } ] ;
Literal       = INT | FLOAT | BOOL | STRING ;

LogicOr       = LogicAnd { "||" LogicAnd } ;
LogicAnd      = Equality { "&&" Equality } ;
Equality      = Rel { ( "==" | "!=" ) Rel } ;
Rel           = Add { ( "<" | "<=" | ">" | ">=" ) Add } ;
Add           = Mul { ( "+" | "-" ) Mul } ;
Mul           = Unary { ( "*" | "/" ) Unary } ;
Unary         = [ "-" | "!" ] Primary ;

Primary       =
      "(" Expr ")"
    | Literal
    | IDENT
    | Call
    | TensorCtor
    | ArrayCtor
    | FieldAccess
    ;

Call          = IDENT "(" ArgListOpt ")" ;
ArgListOpt    = [ Arg { "," Arg } ] ;
Arg           = Expr | KVArg ;

FieldAccess   = Primary "." IDENT [ "(" ArgListOpt ")" ] ;
TensorCtor    = "Tensor" "(" [ ShapeList | ArrayCtor ] ")" ;
ArrayCtor     = "[" [ Expr { "," Expr } ] "]" ;
```

### Operator precedence (lowest → highest)

```
1. ||
2. &&
3. == !=
4. < <= > >=
5. + -
6. * /
7. unary - !
8. field access ., call ()
```