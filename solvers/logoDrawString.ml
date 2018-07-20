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
  let pretty = try Sys.argv.(5) = "pretty" with _ -> false in
  let b0 = Bigarray.(Array1.create int8_unsigned c_layout (8*8)) in
  Bigarray.Array1.fill b0 0 ;
  try
    match parse_program str with
      | Some(p) ->
          let p = analyze_lazy_evaluation p in
          let turtle = run_lazy_analyzed_with_arguments p [] in
          let c = (eval_turtle turtle) in
          let c' = (eval_normal_turtle turtle) in
          let bx = canvas_to_1Darray c 8 in
          if bx = b0 then ()
          else begin
            if sizeFile > 0 then begin
              output_canvas_png ~pretty c sizeFile (fname^".png") ;
              output_canvas_png ~pretty c' sizeFile (fname^"_norm.png")
            end ;
            if size > 0 then npp (canvas_to_1Darray c size)
          end
      | _ -> ()
    with Invalid_argument _ | Failure _ | Stack_overflow -> ()
