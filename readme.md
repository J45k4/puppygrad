# Puppygrad

## Grammar

```
letter        = "A"… "Z" | "a"… "z" | "_" ;
digit         = "0"… "9" ;
hexDigit      = digit | "A"…"F" | "a"…"f" ;

IDENT         = letter , { letter | digit } ;
INT           = digit , { digit } ;
FLOAT         = INT , "." , { digit } , [ exponent ]
              | "." , digit , { digit } , [ exponent ] ;
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

TensorDecl    = "Tensor" IDENT "(" TensorSpecListOpt ")" [ "," AttrList ] [ ";" ] ;
TensorSpecListOpt = [ TensorSpec { "," TensorSpec } ] ;
TensorSpec    = Dim | Attr ;
AttrList      = Attr { "," Attr } ;
Attr          = IDENT "=" AttrValue ;
AttrValue     = INT | FLOAT | BOOL | STRING ;

FuncDecl      = ["export"] "fn" IDENT "(" ParamListOpt ")" Block ;
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

LetStmt       = "let" Binding "=" Expr ;
Binding       = IDENT | TupleBinding ;
TupleBinding  = "(" IDENT { "," IDENT } ")" ;

AssignStmt    = IDENT AssignOp Expr ;
AssignOp      = "=" | "+=" | "-=" | "*=" | "/=" ;

ExprStmt      = Expr ;
ReturnStmt    = "return" Expr ;

ForStmt       = "for" ForHead Block ;
ForHead       = IDENT "in"
    ( "range" "(" CallArgListOpt ")"
    | INT ".." INT
    | "(" IDENT ")"
    ) ;

CallArgListOpt = [ CallArg { "," CallArg } ] ;
CallArg       = Expr | IDENT "=" Expr ;

Expr          =
      IfExpr
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

Unary         = [ "-" | "!" ] Unary | Postfix ;
Postfix       = Primary { PostfixSuffix } ;
PostfixSuffix =
      "(" CallArgListOpt ")"
    | "." IDENT [ "(" CallArgListOpt ")" ]
    | "[" IndexSpec "]"
    ;
IndexSpec     = Expr [ ":" Expr ] ;

Primary       =
      "(" Expr ")"
    | Literal
    | IDENT
    | TensorCtor
    | ArrayCtor
    ;

TensorCtor    = "Tensor" "(" [ TensorCtorBody ] ")" ;
TensorCtorBody = ArrayCtor | ShapeList ;

ShapeList     = Dim { "," Dim } ;
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