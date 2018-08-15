open GeomLib
open Interpreter
open Lexing

exception MalformedProgram of string

let empty28 = Bigarray.(Array1.create int8_unsigned c_layout (28*28))
(*let empty32 = Bigarray.(Array1.create int8_unsigned c_layout (32*32))*)
(*let empty64 = Bigarray.(Array1.create int8_unsigned c_layout (64*64))*)
(*let empty256 = Bigarray.(Array1.create int8_unsigned c_layout (256*256))*)

let _ = Bigarray.Array1.fill empty28 0 ; Random.self_init ()


let npp data =
  for i = 0 to (Bigarray.Array1.dim data) - 2 do
    print_int (data.{i}) ; print_char ',' ;
  done ;
  print_int (data.{((Bigarray.Array1.dim data) - 1)})

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
  let output_img = Sys.argv.(1) in
  let noise = (String.equal Sys.argv.(2) "noise") in
  let program_string = Sys.argv.(3) in
  (try
    (match read_program program_string with
      | Some (program) ->
          (try
            let c  = interpret ~noise program       in
            let l  = Plumbing.canvas_to_tlist 28 c  in
            if (l = empty28) then ()
            else begin
              if not (output_img = "none") then
                Renderer.output_canvas_png c 512 output_img
              else npp l
            end
          with Interpreter.MalformedProgram _ ->
            print_newline ()
            )
      | None -> ())
    with MalformedProgram(error_message) ->
      Printf.printf "%s\n" error_message
    )
