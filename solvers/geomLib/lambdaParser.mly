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
      { Plumbing.var_unit }
  | V_N ; v = var
      { Plumbing.var_next v }
  | V_P ; v = var
      { Plumbing.var_prev v }
  | V_D ; v = var
      { Plumbing.var_double v }
  | V_H ; v = var
      { Plumbing.var_half v }
  | V_O ; v = var
      { Plumbing.var_opposite v }

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
      { Plumbing.concat e1 e2 }
  | EMBED ; e = expr
      { Plumbing.embed e }
  | TURN ; s = somev
      { Plumbing.turn s }
  | REPEAT ; s = somev ; e = expr
      { Plumbing.repeat s e }
  | INTEGRATE ;
    v1 = somev ;
    b  = someb ;
    v2 = somev ;
    v3 = somev ;
    (*v4 = somev ;*)
    (*v5 = somev ;*)
      { Plumbing.integrate v1 b v2 v3 }
