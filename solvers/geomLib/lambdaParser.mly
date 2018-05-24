%token LP
%token RP
%token RUN
%token EMBED
%token TURN
%token REPEAT
%token NOTHING
%token DEFINE
%token JUST
%token INTEGRATE
%token LINE
%token CONCAT
%token NAME
%token V_U
%token V_2
%token V_3
%token V_N
%token V_P
%token V_H
%token V_D
%token V_Di
%token V_O
%token TRUE
%token FALSE
%token EOF

%start <(Interpreter.shapeprogram) option> program
%%
program:
    | EOF
        { None     }
    | LP ; RUN ; e = expr ; RP ; EOF
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
  | V_2
      { Plumbing.var_two }
  | V_3
      { Plumbing.var_three }
  | NAME
      { Plumbing.var_name }
  | V_N ; v = var
      { Plumbing.var_next v }
  | V_P ; v = var
      { Plumbing.var_prev v }
  | V_D ; v = var
      { Plumbing.var_double v }
  | V_Di ; v = var ; v2 = var
      { Plumbing.var_divide v v2 }
  | V_H ; v = var
      { Plumbing.var_half v }
  | V_O ; v = var
      { Plumbing.var_opposite v }

someb:
  | LP ; b = someb ; RP
      { b }
  (*| NOTHING*)
      (*{ None }*)
  | TRUE
      { true }
  | FALSE
      { false }

expr:
  | LP ; e = expr ; RP
      { e }
  | CONCAT ; e1 = expr ; e2 = expr
      { Plumbing.concat e1 e2 }
  | DEFINE ; v = var ;
      { Plumbing.define v }
  | LINE
      { Plumbing.basic_line }
  | EMBED ; e = expr
      { Plumbing.embed e }
  | TURN ; s = somev
      { Plumbing.turn s }
  | REPEAT ; v = somev ; e = expr
      { Plumbing.repeat v e }
  | INTEGRATE ;
    v1 = somev ;
    b  = someb ;
    v2 = somev ;
    v3 = somev ;
    (*v4 = somev ;*)
    (*v5 = somev ;*)
      { Plumbing.integrate v1 b v2 v3 }
