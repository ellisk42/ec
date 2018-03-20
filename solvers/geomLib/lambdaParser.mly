%token LP
%token RP
%token RUN
%token EMBED
%token TURN
%token REPEAT
%token NOTHING
%token JUST
%token INTEGRATE
%token CONCAT
%token V_U
%token V_N
%token V_P
%token V_H
%token V_D
%token V_O
%token TRUE
%token FALSE
%token EOF

%start <(Interpreter.shapeprogram) option> program
%%
program:
    | EOF
        { None     }
    | LP ; RUN ; LP ; e = expr ; RP ; RP ; EOF
        { Some (e) }

somev:
  | LP ; s = somev ; RP
      { s }
  | NOTHING
      { None }
  | JUST ; v = var
      { Some (v) }
var:
  | LP ; v = var ; RP
      { v }
  | V_U
      { Interpreter.Unit }
  | V_N ; v = var
      { Interpreter.Next (v) }
  | V_P ; v = var
      { Interpreter.Prev (v) }
  | V_D ; v = var
      { Interpreter.Double (v) }
  | V_H ; v = var
      { Interpreter.Half (v) }
  | V_O ; v = var
      { Interpreter.Opposite (v) }

someb:
  | LP ; b = someb ; RP
      { b }
  | NOTHING
      { None }
  | JUST ; TRUE
      { Some(true) }
  | JUST ; FALSE
      { Some(false) }

expr:
  | LP ; e = expr ; RP
      { e }
  | CONCAT ; e1 = expr ; e2 = expr
      { Interpreter.Concat (e1, e2) }
  | EMBED ; e = expr
      { Interpreter.Embed (e) }
  | TURN ; s = somev
      { Interpreter.Turn (s) }
  | REPEAT ; s = somev ; e = expr
      { Interpreter.Repeat (s, e) }
  | INTEGRATE ;
    v1 = somev ;
    b  = someb ;
    v2 = somev ;
    v3 = somev ;
    v4 = somev ;
    v5 = somev ;
      { Interpreter.Integrate (v1,b,(v2,v3,v4,v5)) }
