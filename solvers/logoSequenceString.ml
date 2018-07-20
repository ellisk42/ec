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
  let str      = Sys.argv.(1)
  and folder   = Sys.argv.(2) in
  try
    match parse_program str with
      | Some(p) ->
          prerr_endline "Parsed" ;
          let p = analyze_lazy_evaluation p in
          prerr_endline "Parsed" ;
          let turtle = run_lazy_analyzed_with_arguments p [] in
          let c = eval_turtle ~sequence:(folder^"/output_") turtle in
          prerr_endline "evaled" ;
          output_canvas_png c 512 (folder^".png") ;
          prerr_endline "drawn"
      | _ -> ()
    with Invalid_argument _ | Failure _ | Stack_overflow -> ()
