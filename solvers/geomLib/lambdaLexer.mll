{
  open Lexing
  open LambdaParser
  exception SyntaxError of string
}

let white = [' ' '\t' '\n' '\r']+

rule read =
  parse
  | white          { read lexbuf }
  | '#'            { read lexbuf } (* Completely ignore this symbol [1] *)
  | '('            { LP          }
  | ')'            { RP          }
  | "run"          { RUN         }
  | "embed"        { EMBED       }
  | "turn"         { TURN        }
  | "repeat"       { REPEAT      }
  | "nothing"      { NOTHING     }
  | "define"       { DEFINE      }
  | "just"         { JUST        }
  | "integrate"    { INTEGRATE   }
  | "concat"       { CONCAT      }
  | "basic_line"   { LINE        }
  | "var_unit"     { V_U         }
  | "var_two"      { V_2         }
  | "var_three"    { V_3         }
  | "var_name"     { NAME        }
  | "var_next"     { V_N         }
  | "var_prev"     { V_P         }
  | "var_half"     { V_H         }
  | "var_double"   { V_D         }
  | "var_divide"   { V_Di        }
  | "var_opposite" { V_O         }
  | "true"         { TRUE        }
  | "false"        { FALSE       }
  | eof            { EOF         }

(*
 * [1]: It means "EC invented this as a "new" instuction of cost 1" but the new
 * instruction is under this symbol so at execute time we just ingore this.
 *)
