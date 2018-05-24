%token <bool> PEN
%token <string> VAR
%token BEGIN_BLOCK
%token END_BLOCK
%token BEGIN_ARGS
%token END_ARGS
%token COLON
%left COLON
%token NOP
%token INTEGRATE
%token REPEAT
%token EMBED
%token TURN
%token DOUBLE
%token HALF
%token NEXT
%token PREV
%token OPPOSITE
%token DIVIDE
%token UNIT
%token TWO
%token THREE
(*%token INDEFINITE*)
%token COMMA_ARGS
%token ARG_ANGLE
%token ARG_T
%token ARG_PEN
%token ARG_SPEED
%token ARG_ACCEL
%token ARG_ANGULARSPEED
%token ARG_ANGULARACCEL
%token EQUALS
%token EOF

%start <(Interpreter.shapeprogram) option> program
%%
program:
    | EOF       { None }
    | v = value ; EOF { Some (v) }
;

optional_comma:
    | COMMA_ARGS {}
    | {}

expr:
  | UNIT  { Interpreter.Unit }
  | TWO   { Interpreter.Next(Interpreter.Unit) }
  | THREE { Interpreter.Next(Interpreter.Next(Interpreter.Unit)) }
  (*| INDEFINITE { Interpreter.Indefinite }*)
  | DOUBLE ; BEGIN_ARGS ; e = expr ; END_ARGS {Interpreter.Double (e) }
  | HALF ; BEGIN_ARGS ; e = expr ; END_ARGS {Interpreter.Half (e) }
  | NEXT ; BEGIN_ARGS ; e = expr ; END_ARGS {Interpreter.Next (e) }
  | DIVIDE ; BEGIN_ARGS ; e1 = expr ; COMMA_ARGS ; e2 = expr ; END_ARGS
    {Interpreter.Divide (e1,e2) }
  | PREV ; BEGIN_ARGS ; e = expr ; END_ARGS {Interpreter.Prev (e) }
  | OPPOSITE ; BEGIN_ARGS ; e = expr ; END_ARGS {Interpreter.Opposite (e) }
  | s = VAR { Interpreter.Name s }

optional_turn_args:
    | BEGIN_ARGS ; ARG_ANGLE ; EQUALS ; e = expr ; END_ARGS {Some e}
    | BEGIN_ARGS ; END_ARGS {None}
    | {None}
turn:
    | TURN ; args = optional_turn_args ; {Interpreter.Turn args}


optional_repeat_args:
    | BEGIN_ARGS ; e = expr ; END_ARGS { Some e }
    | {None}
repeat:
    | REPEAT ; n = optional_repeat_args ; BEGIN_BLOCK ; p = value ; END_BLOCK
        {Interpreter.Repeat (n,p)}


optional_integrate_args:
    | BEGIN_ARGS ;
        d = optional_integrate_d ; optional_comma ;
        pen = optional_integrate_pen ; optional_comma ;
        speed = optional_integrate_speed ; optional_comma ;
        accel = optional_integrate_accel ; optional_comma ;
        angularSpeed = optional_integrate_angularSpeed ; optional_comma ;
        angularAccel = optional_integrate_angularAccel ;
        END_ARGS
        {Interpreter.Integrate (d,pen,(speed,accel,angularSpeed,angularAccel))}
    | { Interpreter.Integrate (None,None,(None,None,None,None)) }
optional_integrate_d:
    | ARG_T ; EQUALS ; e = expr { Some e }
    | {None}
optional_integrate_pen:
    | ARG_PEN ; EQUALS ; b = PEN { Some b }
    | {None}
optional_integrate_speed:
    | ARG_SPEED ; EQUALS ; e = expr { Some e }
    | {None}
optional_integrate_accel:
    | ARG_ACCEL ; EQUALS ; e = expr { Some e }
    | {None}
optional_integrate_angularSpeed:
    | ARG_ANGULARSPEED ; EQUALS ; e = expr { Some e }
    | {None}
optional_integrate_angularAccel:
    | ARG_ANGULARACCEL ; EQUALS ; e = expr { Some e }
    | {None}
integrate:
    | INTEGRATE ;
        i = optional_integrate_args { i }

value:
    | t = turn { t }
    | EMBED ; BEGIN_BLOCK ; p = value ; END_BLOCK {Interpreter.Embed p}
    | p1 = value ; COLON ; p2 = value {Interpreter.Concat (p1,p2)}
    | s = VAR ; EQUALS ; e = expr {Interpreter.Define (s,e)}
    | NOP { Interpreter.Nop }
    | r = repeat { r }
    | i = integrate { i }
