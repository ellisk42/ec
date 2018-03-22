open GeomLib
open Interpreter
open Lexing

exception MalformedProgram of string

let npp data =
  for i = 0 to (Bigarray.Array1.dim data) - 1 do
    print_int (data.{i})
  done

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
            let c = interpret program in
            let l1 = Plumbing.canvas_to_tlist 32 c in
            let l2 = Plumbing.canvas_to_tlist 64 c in
            (npp l1 ; print_newline () ;
             npp l2 ; print_newline ())
          with _ ->
            (print_string (String.make (32*32) '0') ; print_newline () ;
             print_string (String.make (64*64) '0') ; print_newline ())
            )
      | None -> ())
    with MalformedProgram(error_message) ->
      Printf.printf "%s\n" error_message
    )
