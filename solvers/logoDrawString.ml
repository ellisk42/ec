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

let smooth_logo_wrapper t2t k s0 =
  let e = 1./.20. in
  let (p,s) = t2t k s0 in
  let rec smooth_path command = match command with
    | SEGMENT(x1,y1,x2,y2) ->
      let dx = x2-.x1 in
      let dy = y2-.y1 in
      let l = dx*.dx+.dy*.dy |> sqrt in
      if l <= e then [command] else
        let f = e/.l in
        let x = x1 +. f*.dx in
        let y = y1 +. f*.dy in        
        (SEGMENT(x1,y1,x,y)) :: smooth_path (SEGMENT(x,y,x2,y2))
  in       
  (p |> List.map smooth_path |> List.concat, s)

let _ =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let j = Yojson.Basic.from_channel Pervasives.stdin in
  let programs = member "programs" j in
  let programs = to_list programs in
  let programs = List.map to_string programs in
  let programs = List.map parse_program programs in

  let pretty = try
      to_bool (member "pretty" j)
    with _ -> false
  in
  let smooth_pretty = try
      to_bool (member "smoothPretty" j)
    with _ -> false
  in
  let export_size = try
    with _ -> 0
  in
  

  let sizeFile = int_of_string (Sys.argv.(1))
  and fname    = Sys.argv.(2)
  and size     = int_of_string (Sys.argv.(3))
  and str      = Sys.argv.(4) in
  let smooth_pretty = try Sys.argv.(5) = "smooth_pretty" with _ -> false in
  let pretty = smooth_pretty || try Sys.argv.(5) = "pretty" with _ -> false in
  let b0 = Bigarray.(Array1.create int8_unsigned c_layout (8*8)) in
  Bigarray.Array1.fill b0 0 ;
  try
    match parse_program str with
      | Some(p) ->
          let p = analyze_lazy_evaluation p in
          let turtle = run_lazy_analyzed_with_arguments p [] in
          let turtle = if smooth_pretty then smooth_logo_wrapper turtle else turtle in
          let c = (eval_turtle turtle) in
          let bx = canvas_to_1Darray c 8 in
          if bx = b0 then ()
          else begin
            if sizeFile > 0 then begin
              output_canvas_png ~pretty c sizeFile (fname^".png") ;
            end ;
            if size > 0 then npp (canvas_to_1Darray c size)
          end
      | _ -> ()
    with Invalid_argument _ | Failure _ | Stack_overflow -> ()
