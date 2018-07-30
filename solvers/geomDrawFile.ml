open GeomLib
open Plotter
open Renderer
open Interpreter
open Printf
open Lexing

exception MalformedProgram of string

let _ = Random.self_init ()

let print_pos lexbuf = 
  let pos = lexbuf.lex_curr_p in
  sprintf "(line %d ; column %d)"
          pos.pos_lnum (pos.pos_cnum - pos.pos_bol)

let parse_with_error lexbuf =
  try GeomParser.program GeomLexer.read lexbuf with
  | GeomLexer.SyntaxError msg ->
      let pos_string = print_pos lexbuf in
      raise (MalformedProgram
                (sprintf "Error at position %s, %s" pos_string msg))
  | GeomParser.Error ->
      let pos_string = print_pos lexbuf in
      raise (MalformedProgram (sprintf "Error at position %s\n" pos_string))

let read_program program_string =
  let lexbuf = Lexing.from_string program_string in
  let program = parse_with_error lexbuf in
  program

let file_to_string filename =
  let ic = open_in filename in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic ;
  s

let _ =
  if (Array.length Sys.argv != 2) then failwith "You need to provide exactly one argument, namely the name of the .LoG file you want to parse and execute, and this program will output a .png file accordingly." ;
  let program_string = file_to_string  Sys.argv.(1) in
  (try
    (match read_program program_string with
      | Some (program) ->
          let canvas = interpret program
          and canvas_norm = interpret_normal program
          and prefix = Filename.chop_suffix Sys.argv.(1) ".LoG" in
          (try Unix.mkdir prefix 0o777
          with Unix.Unix_error(Unix.EEXIST, _, _) -> ()) ;
          let pngFName = (prefix^"/output_l.png")
          and pngFNameh= (prefix^"/output_h.png")
          and pngFNamevh= (prefix^"/output_vh.png")
          in
          output_canvas_png canvas 28 pngFName ;
          output_canvas_png canvas 128 pngFNameh ;
          output_canvas_png canvas 512 pngFNamevh ;
          for i = 0 to 30 do
            let canvas = interpret ~noise:true program in
            let pngRName = (prefix^"/random_"^(string_of_int i)^".png") in
            output_canvas_png canvas 512 pngRName
          done
      | None -> ())
    with MalformedProgram(error_message) ->
      Printf.printf "%s\n" error_message
    )
