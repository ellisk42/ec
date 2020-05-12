open Core

open LogoLib.VGWrapper

open Timeout
open Task
open Utils
open Program
open Type


let tstroke = make_ground "tstroke";;
let tangle = make_ground "tangle";;
let tscale = make_ground "tscale";;
let tdistance = make_ground "tdist";;
let tmatrix = make_ground "ttransmat";;
let trep = make_ground "trep";;
let torder = make_ground "ttorder";;

exception OutsideOfCanvas;;


type coordinate = float*float

type shape = 
  | Line of coordinate*coordinate

let matrix_multiply x y =
  let x0 = Array.length x
  and y0 = Array.length y in
  let y1 = if y0 = 0 then 0 else Array.length y.(0) in
  let z = Array.make_matrix x0 y1 0. in
  for i = 0 to x0-1 do
    for j = 0 to y1-1 do
      for k = 0 to y0-1 do
        z.(i).(j) <- z.(i).(j) +. x.(i).(k) *. y.(k).(j)
      done
    done
  done;
  z

let (@@) a b = matrix_multiply a b;;

let ( *& ) m (x,y) =
  let v = m @@ [|[|x|];[|y|];[|1.|];|] in
  v.(0).(0), v.(1).(0)


let rotate_matrix t =
  let cos = cos t in
  let sin = sin t in
  [| [| cos; -.sin; 0.|];
     [| sin;   cos; 0. |];
     [|0.; 0.; 1.|] |]

let translate_matrix x y =
  [| [| 1.;0.;x|];
     [|0.;1.;y|];
     [|0.;0.;1.;|] |]

let scale_matrix s =
  [|[|s; 0.; 0.|]; [|0.; s; 0.|]; [|0.; 0.; 1.|]|]




              
(* a drawing routine is a list of curves *)
type drawing_routine = shape list;;

ignore(primitive "connect" (tstroke @> tstroke @> tstroke) (@));;
ignore(primitive "emptystroke" tstroke []);;
ignore(primitive "emptystrokeC" (tstroke @> tstroke) (fun s -> s));;
ignore(primitive "line" tstroke [Line((0.,0.),(1.,0.))]);;
ignore(primitive "lineC" (tstroke @> tstroke) (fun s -> (Line((0.,0.),(1.,0.))) :: s));;
let circle_strokes = (
    (* Python creates 30 equally spaced vertices around the circle *)
    List.range ~start:`inclusive ~stop:`exclusive
      0 29 |> 
    List.map ~f:(fun i ->
        let t = 2.*.pi*.(Float.of_int i)/.29. in
        let t' = 2.*.pi*.(Float.of_int (i+1))/.29. in
        let p1 = (0.5*.cos t, 0.5*.sin t) in
        let p2 = (0.5*.cos t', 0.5*.sin t') in
        Line(p1,p2)));;
ignore(primitive "circle" tstroke circle_strokes);;
ignore(primitive "circleC" (tstroke @> tstroke) (fun s -> s @ circle_strokes));;
ignore(primitive "transmat" (tmaybe tscale @> tmaybe tangle @> tmaybe tdistance @> tmaybe tdistance @> tmaybe torder @> tmatrix)
         (fun scale angle d1 d2 o -> 
            let scale = match scale with
                None -> 1. | Some(s) -> s
            in
            let angle = match angle with | None -> 0. | Some(a) -> a in
            let x = match d1 with | None -> 0. | Some(x) -> x in
            let y = match d2 with | None -> 0. | Some(x) -> x in
            let o = match o with | None -> "trs" | Some(o) -> o in

            let t = translate_matrix x y in
            let r = rotate_matrix angle in
            let s = scale_matrix scale in
            if o = "trs" then
              t@@r@@s
            else if o = "tsr" then
              t@@s@@r
            else if o = "rts" then
              r@@t@@s
            else if o = "rst" then
              r@@s@@t
            else if o = "srt" then
              s@@r@@t
            else if o = "str" then
              s@@t@@r
            else
              assert (false)
         ));;

let transform s m = s |> List.map ~f:(fun s -> match s with
    | Line(u,v) -> Line(m *& u, m*& v));;

ignore(primitive "transform" (tstroke @> tmatrix @> tstroke) transform);;
ignore(primitive "transformC" ((tstroke @> tstroke) @> tmatrix @> tstroke @> tstroke)
         (fun s m k ->
            transform (s []) m @ k));;
ignore(primitive "None" (tmaybe t0) None);;
ignore(primitive "Some" (t0 @> tmaybe t0) (fun x -> Some(x)));;

(* 
([-2.; -1.5; -1.; -0.5; -0.25; 0.; 0.25; 0.5; 1.; 1.5; 2.]@
 (List.range ~stop:`exclusive 3 7 |> List.map ~f:(fun n ->
      let n = Float.of_int n in
    0.5/.(tan (pi/.n)))))|> List.iteri ~f:(fun i d ->
    ignore(primitive (Printf.sprintf "dist%d" i) tdistance d));;
[0.5; 1.; 1.25; 1.5; 2.; 2.5; 3.; 4.] |> List.iteri ~f:(fun i d ->
    ignore(primitive (Printf.sprintf "scale%d" i) tscale d));;
["trs"; "tsr"; "rts"; "rst"; "srt"; "str"] |> List.iter ~f:(fun o ->
    ignore(primitive o torder o));;
[0;1;2;3;4;5;6;] |> List.iter ~f:(fun n ->
    ignore(primitive (Printf.sprintf "rep%d" n) trep n));;
((List.range ~stop:`exclusive 0 8 |> List.map ~f:(fun j -> (Float.of_int j)*.2.*.pi/.8.)) @ [-.2.*.pi/.6.;-.2.*.pi/.12.]) |> List.iteri ~f:(fun i t ->
    ignore(primitive (Printf.sprintf "angle%d" i) tangle t));; *)

([-2.5; -2.; -1.5; -1.; -0.5; -0.25; 0.; 0.25; 0.5; 1.; 1.5; 2.; 2.5; 3.; -1.75; -0.65; 0.45; 1.55; 1.1]@
 (List.range ~stop:`exclusive 3 7 |> List.map ~f:(fun n ->
      let n = Float.of_int n in
    0.5/.(tan (pi/.n)))))|> List.iteri ~f:(fun i d ->
    ignore(primitive (Printf.sprintf "dist%d" i) tdistance d));;
[0.5; 1.; 1.25; 1.5; 2.; 2.5; 3.; 4.] |> List.iteri ~f:(fun i d ->
    ignore(primitive (Printf.sprintf "scale%d" i) tscale d));;
["trs"; "tsr"; "rts"; "rst"; "srt"; "str"] |> List.iter ~f:(fun o ->
    ignore(primitive o torder o));;
[0;1;2;3;4;5;6;] |> List.iter ~f:(fun n ->
    ignore(primitive (Printf.sprintf "rep%d" n) trep n));;
((List.range ~stop:`exclusive 0 8 |> List.map ~f:(fun j -> (Float.of_int j)*.2.*.pi/.8.)) @ [-.2.*.pi/.6.;-.2.*.pi/.12.]) |> List.iteri ~f:(fun i t ->
    ignore(primitive (Printf.sprintf "angle%d" i) tangle t));;

ignore(primitive "dist98" tdistance -0.2);;
ignore(primitive "dist99" tdistance -0.2);;


let reflect_implementation =
  (fun p t ->
     let t' = t -. pi/.2. in
     let first_rotation = rotate_matrix (-.t') in
     let second_rotation = rotate_matrix t' in
     let reflect = [| [|-1.;0.;0.;|];
                      [|0.;1.;0.;|];
                      [|0.;0.;1.;|]|]
     in
     let m = second_rotation @@ reflect @@ first_rotation in
     transform p m);;
ignore(primitive "reflect" (tstroke @> tangle @> tstroke)
         reflect_implementation);;
ignore(primitive "reflectC" ((tstroke @> tstroke) @> tangle @> tstroke @> tstroke)
         (fun s a k ->
            reflect_implementation (s []) a @ k));;       

let repeat_implementation = (fun p n m ->
            List.fold_right (List.range ~stop:`exclusive 0 n) ~init:[p]
              ~f:(fun _ (s :: ss) ->
                  transform s m :: s :: ss) |>
            List.concat);;
ignore(primitive "repeat" (tstroke @> trep @> tmatrix @> tstroke)
         repeat_implementation);;
ignore(primitive "repeatC" ((tstroke @> tstroke) @> trep @> tmatrix @> tstroke @> tstroke)
         (fun s r m k ->
            repeat_implementation (s []) r m @ k));;

   
                                    


let render_canvas ?length:(length=6.) (dr : drawing_routine)  =
  try
    let desired_size = 9. in (* the backend expects a 9x9 render *)
    let check c =
      if c < 0. || c > desired_size then raise OutsideOfCanvas
    in 
    let rec make_canvas = function
      | [] -> new_canvas()
      | Line((x1,y1),(x2,y2)) :: rest ->
        let x1 = (x1+.length/.2.)*.desired_size/.length in
        let y1 = (y1+.length/.2.)*.desired_size/.length in
        let x2 = (x2+.length/.2.)*.desired_size/.length in
        let y2 = (y2+.length/.2.)*.desired_size/.length in
        check x1; check x2; check y1; check y2;
        let suffix = make_canvas rest in
        lineto (moveto suffix x1 y1) x2 y2
        (* | Circle({x;y;r}) :: rest -> *)
        (*   circle (make_canvas rest) x y *)
    in 
    Some(make_canvas dr)
  with OutsideOfCanvas -> None;;

let drawing_cost (dr : drawing_routine) : float =
  dr |> List.fold_left ~init:0. ~f:(fun cost_so_far k ->
      match k with
      | Line((x,y),(x',y')) ->
        let dx = x-.x' in
        let dy = y-.y' in
        let d2 = dx*.dx+.dy*.dy in
        sqrt d2 +. cost_so_far);;
      


let recent_draw_program : (program*((((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t)*float) option)) option ref = ref None;;
let run_recent_draw ?attempts:(attempts=1) ~timeout ~resolution p =
  (* this allows us to handle both continuation passing and also not continuation passing *)
  let p = match p with
    | Abstraction(_) -> Apply(p,primitive_empty)
    | _ -> p
  in
  
  match !recent_draw_program with
  | Some(p',y) when program_equal p p' -> y
  | _ ->
    let y = 
      run_for_interval ~attempts:attempts timeout
        (fun () ->
           let p = analyze_lazy_evaluation p in
           let x = run_lazy_analyzed_with_arguments p [] in
           let length = drawing_cost x in 
           match render_canvas x with
           | Some(c) -> Some((length, canvas_to_1Darray c resolution))
           | None -> None)
    in
    let y = match y with
      | Some(Some((l,y'))) -> Some((y',l))
      | Some(None) -> None (* outside of canvas *)
      | None -> None (* timeout *)
    in
    recent_draw_program := Some(p,y);
    y
;;

register_special_task "draw" (fun  extras ?timeout:(timeout = 0.001) name ty examples ->
    let open Yojson.Basic.Util in
    let spec = 
      extras |> member "trajectory" |> to_list |> List.map ~f:(fun l ->
          let [p1;p2] = l |> to_list in
          let [x1;y1] = p1 |> to_list in
          let [x2;y2] = p2 |> to_list in
          Line((x1 |> to_float, y1 |> to_float),
               (x2 |> to_float, y2 |> to_float)))
    in

    let bounded_cost = try
        extras |> member "bounded_cost" |> to_bool
      with _ -> false
    in
    let l2 = try
        Some(extras |> member "l2" |> to_float)
      with _ -> None
    in

    let resolution = 28 in

    let spec_cost = drawing_cost spec in

    let spec = match render_canvas spec with
      | None ->
        Printf.eprintf "FATAL: task %s draws outside of canvas\n" name;
        assert (false)
      | Some(spec') -> spec'
    in
    let spec = canvas_to_1Darray spec resolution in

    {name = name;
     task_type = ty;
     log_likelihood =
       (fun program ->
          let yh = run_recent_draw ~timeout ~resolution program in
          match yh with
          | None -> log 0.
          | Some(yh,l) -> begin
              match l2 with
              | None ->
                if (LogoLib.LogoInterpreter.fp_equal spec yh 5) &&
                   ((not bounded_cost) || l <= spec_cost*.1.1)
                then 0.
                else log 0.
              | Some(coefficient) ->
                let d = (LogoLib.LogoInterpreter.distance spec yh) -. 50. in
                (0.-.coefficient)*.d
            end)
    })
