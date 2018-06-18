open LogoLib
open LogoInterpreter
open VGWrapper

open Differentiation
open Program

let npp data =
  for i = 0 to (Bigarray.Array1.dim data) - 2 do
    print_int (data.{i}) ; print_char ',' ;
  done ;
  print_int (data.{((Bigarray.Array1.dim data) - 1)}) ;
  print_newline ()

let _ =
  let sizeFile = int_of_string (Sys.argv.(1))
  and fname    = Sys.argv.(2)
  and size     = int_of_string (Sys.argv.(3))
  and str      = Sys.argv.(4) in
  let b0 = Bigarray.(Array1.create int8_unsigned c_layout (size*size)) in
  Bigarray.Array1.fill b0 0 ;
  try
    match parse_program str with
      | Some(p) ->
          let p = analyze_lazy_evaluation p in
          let turtle = run_lazy_analyzed_with_arguments p [] in
          (*pp_turtle turtle ;*)
          let c = (eval_turtle turtle) in
          (*prerr_endline (Vg.P.to_string c) ;*)
          let bx = canvas_to_1Darray c size in
          if bx = b0 then prerr_endline "emptyDrawing"
          else begin
            output_canvas_png c sizeFile fname ;
            npp (canvas_to_1Darray c size)
          end
      | _ ->
          (prerr_endline "Could not parse")
    with Invalid_argument _ | Failure _ | DIV0 | Stack_overflow ->
      (prerr_endline "other error")
