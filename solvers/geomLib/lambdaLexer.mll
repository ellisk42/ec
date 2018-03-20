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
  | "just"         { JUST        }
  | "integrate"    { INTEGRATE   }
  | "concat"       { CONCAT      }
  | "var_unit"     { V_U         }
  | "var_next"     { V_N         }
  | "var_prev"     { V_P         }
  | "var_half"     { V_H         }
  | "var_double"   { V_D         }
  | "var_opposite" { V_O         }
  | "true"         { TRUE        }
  | "false"        { FALSE       }
  | eof            { EOF         }

(*
 * [1]: It means "EC invented this as a "new" instuction of cost 1" but the new
 * instruction is under this symbol so at execute time we just ingore this.
 *)
