open GeomLib
open Interpreter
open Lexing

exception MalformedProgram of string

let rec npp l = match l with
 | []     -> ()
 | x :: r ->
     if (x = 0) then print_char '0' else print_char '1' ;
     npp r

let print_pos lexbuf = 
  let pos = lexbuf.lex_curr_p in
  Printf.sprintf "(line %d ; column %d)"
          pos.pos_lnum (pos.pos_cnum - pos.pos_bol)

let parse_with_error lexbuf =
  try LambdaParser.program LambdaLexer.read lexbuf with
  | LambdaLexer.SyntaxError msg ->
      let pos_string = print_pos lexbuf in
      raise (MalformedProgram
                (Printf.sprintf "Error at position %s, %s" pos_string msg))
  | LambdaParser.Error ->
      let pos_string = print_pos lexbuf in
      raise (MalformedProgram (Printf.sprintf "Error at position %s\n" pos_string))

let read_program program_string =
  try
    let lexbuf = Lexing.from_string program_string in
    let program = parse_with_error lexbuf in
    program
  with e -> (print_endline program_string ; raise e)

let _ =
  let program_string = Sys.argv.(1) in
  (try
    (match read_program program_string with
      | Some (program) ->
          (try
            let l = Plumbing.run program in
            (npp l ; print_newline ())
          with _ ->
            (print_string (String.make (16*16) '0') ;
            print_newline ())
            )
      | None -> ())
    with MalformedProgram(error_message) ->
      Printf.printf "%s\n" error_message
    )
