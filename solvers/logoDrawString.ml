open Core

open LogoLib
open LogoInterpreter
open VGWrapper

open Differentiation
open Program

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
  (p |> List.map  ~f:smooth_path |> List.concat, s)


let _ =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel Pervasives.stdin in
  let open Yojson.Basic in
  let open Utils in
  let open Timeout in
  let jobs = to_list (member "jobs" j) in
  
  let pretty = try
      to_bool (member "pretty" j)
    with _ -> false
  in
  let smooth_pretty = try
      to_bool (member "smoothPretty" j)
    with _ -> false
  in
  let timeout = try
      to_float (member "timeout" j)
    with _ -> 0.01
  in

  let trim s =
    if s.[0] = '"' then String.sub s 1 (String.length s - 2) else s
  in 

  let b0 = Bigarray.(Array1.create int8_unsigned c_layout (8*8)) in
  Bigarray.Array1.fill b0 0 ;
  let results = List.map jobs ~f:(fun j ->
      let size = to_int (member "size" j) in
      let export = try
          match to_string (member "export" j) with
          | "null" -> None
          | e -> Some(trim e)
        with _ -> None
      in

      let animate = try
          to_bool (member "animate" j)
        with _ -> false
      in 
      
      let p = to_string (member "program" j) |> trim in
      let p = safe_get_some (Printf.sprintf "Could not parse %s\n" p) (parse_program p) in
      if animate then
        match export with
        | None -> assert (false)
        | Some(export) -> 
          let p = analyze_lazy_evaluation p in
          let turtle = run_lazy_analyzed_with_arguments p [] in
          let turtle = if smooth_pretty then smooth_logo_wrapper turtle else turtle in
          let cs = animate_turtle turtle in
          List.iteri cs (fun j c ->
              output_canvas_png ~pretty c size (Printf.sprintf "%s_%09d.png" export j));
          Sys.command (Printf.sprintf "convert -delay 1 -loop 0 %s_*.png %s.gif"
                         export export);
          Sys.command (Printf.sprintf "rm %s_*.png" export);
          `String("exported")
      else 
        try
          match run_for_interval timeout (fun () ->
              let p = analyze_lazy_evaluation p in
              let turtle = run_lazy_analyzed_with_arguments p [] in
              let turtle = if smooth_pretty then smooth_logo_wrapper turtle else turtle in
              let c = eval_turtle turtle in
              let array = canvas_to_1Darray c size in
              c, array) with
          | None -> `String("timeout")
          | Some(c, array) ->       
            let bx = canvas_to_1Darray c 8 in
            if bx = b0 then `String("empty")
            else
              match export with
              | Some(fn) -> (output_canvas_png ~pretty c size fn;
                             `String("exported"))
              | None ->
                `List(List.map (range (Bigarray.Array1.dim array)) ~f:(fun i -> `Int(array.{i})))
        with _ -> `String("exception")
    )
  in

  print_string (pretty_to_string (`List(results)))
