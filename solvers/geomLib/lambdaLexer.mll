{
  open Lexing
  open LambdaParser
  exception SyntaxError of string
}

let white = [' ' '\t']+

rule read =
  parse
  | white          { read lexbuf }
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
  (*| iden as i  { VAR i }*)
