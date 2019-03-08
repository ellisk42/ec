open VGWrapper

exception DoesNotMatch

type state =
  { x : float
  ; y : float
  ; t : float
  ; p : bool
  }

type logo_instruction = SEGMENT of float*float*float*float

type program = logo_instruction list

type turtle = (state -> (program * state))

let logo_NOP : turtle = fun s -> ([], s)

let init_state () = {x = d_from_origin; y = d_from_origin; t = 0.; p = true}

let flush_everything () =
  Pervasives.flush stdout;
  Pervasives.flush stderr

let center_logo_list (l : logo_instruction list) : logo_instruction list =
  let rec minimum = function
    | [] -> assert (false)
    | [x] -> x
    | h :: t -> min h (minimum t)
  and maximum = function
    | [] -> assert (false)
    | [x] -> x
    | h :: t -> max h (maximum t)
  in 
  match l with
  | [] -> []
  | _ ->
    let xs = l |> List.map (function | (SEGMENT(x,_,x',_)) -> [x-.d_from_origin;
                                                               x'-.d_from_origin;]) |> List.concat in 
    let ys = l |> List.map (function | (SEGMENT(_,y,_,y')) -> [y-.d_from_origin;
                                                               y'-.d_from_origin;]) |> List.concat in
    let x0 = xs |> minimum in
    let x1 = xs |> maximum in
    let y0 = ys |> minimum in
    let y1 = ys |> maximum in
    let dx = (x1-.x0)/.2.+.x0 in
    let dy = (y1-.y0)/.2.+.y0 in
    let d_from_origin = 0. in
    l |> List.map (function | (SEGMENT(x,y,x',y')) ->
        SEGMENT(x-.dx+.d_from_origin, y-.dy+.d_from_origin,
                x'-.dx+.d_from_origin, y'-.dy+.d_from_origin))

let logo_contained_in_canvas (l : logo_instruction list) =
  let okay p = p >= 0.0 && p <= 2.*.d_from_origin in 
  List.for_all (function | (SEGMENT(a,b,c,d)) -> okay a && okay b && okay c && okay d) l


  
let pp_logo_instruction i =
  match i with
  | SEGMENT(x1,y1,x2,y2) -> Printf.eprintf "segment{%f,%f,%f,%f}"
                              x1 y1 x2 y2
                              
let pp_turtle t =
  let l,_ = (t logo_NOP) (init_state ()) in
  List.iter
    (fun e ->
       pp_logo_instruction e ;
       prerr_string "; " ;
       flush_everything () )
    l ;
  prerr_newline ()

let eval_turtle ?sequence (t2t : turtle -> turtle) =
  let p,_ = (t2t logo_NOP) (init_state ()) in
  let p = center_logo_list p in
  let c = ref (new_canvas ()) in
  let lineto x y = (c := (lineto !c x y))
  and moveto x y = (c := (moveto !c x y)) in
  let t = init_state () in
  moveto t.x t.y ;
  let rec eval_instruction i = match i with
    | SEGMENT(x1,y1,x2,y2) ->
      (moveto x1 y1; lineto x2 y2)  in
  List.iter eval_instruction p ;
  !c

let animate_turtle ?sequence (t2t : turtle -> turtle) =
  let p,_ = (t2t logo_NOP) (init_state ()) in
  let p = center_logo_list p in
  let draw_list p = 
    let c = ref (new_canvas ()) in
    let lineto x y = (c := (lineto !c x y))
    and moveto x y = (c := (moveto !c x y)) in
    let t = init_state () in
    moveto t.x t.y ;
    let rec eval_instruction i = match i with
      | SEGMENT(x1,y1,x2,y2) ->
        (moveto x1 y1; lineto x2 y2)  in
    List.iter eval_instruction p ;
    !c
  in
  let rec map_suffixes l f = match l with
    | [] -> []
    | _ :: xs -> f l :: map_suffixes xs f
  in 
  List.rev (map_suffixes (List.rev p) draw_list)
  
let logo_PU : turtle =
  fun s -> ([], {s with p = false})

let logo_PD : turtle =
  fun s -> ([], {s with p = true})

let logo_RT : float -> turtle =
  fun angle -> fun s -> ([], {s with t = s.t +. angle})

let logo_FW : float -> turtle =
  let pi = 4.0 *. atan 1.0 in 
  fun length  ->
  fun s ->
    let x' = s.x +. (length *. cos(s.t*.2.*.pi)) in
    let y' = s.y +. (length *. sin(s.t*.2.*.pi)) in
    let s' = {s with x = x'; y = y';} in
    let k = if s.p then [SEGMENT(s.x,s.y,x',y')] else [] in 
    (k,s')


let logo_SEQ : turtle -> turtle -> turtle =
  fun p1 p2 ->
    fun s ->
      (let l, s' = p1 s in
      let l', s'' = p2 s' in
      (l @ l', s''))

let logo_GET : (state -> turtle) -> turtle =
  fun f ->
    fun s ->
      f s s

(* let logo_SET : (state -> turtle) = fun s -> fun _ -> ([SET({s with t=s.t *. 4. *. atan(1.)})], s) *)
let logo_SET : (state -> turtle) = fun s -> fun _ -> ([], s)

   

let turtle_to_list turtle =
  let l,_ = (turtle logo_NOP) (init_state ()) in l |> center_logo_list

let turtle_to_png turtle resolution filename =
  output_canvas_png (eval_turtle turtle) resolution filename

let turtle_to_array turtle resolution =
  canvas_to_1Darray (eval_turtle turtle) resolution


let turtle_to_both turtle resolution filename =
  let c = (eval_turtle turtle) in
  output_canvas_png c resolution filename ;
  canvas_to_1Darray c resolution

let max = 28. *. 256.

let fp_equal x y eps =
  try
    for i = 0 to Bigarray.Array1.dim x - 1 do
      if (abs (x.{i} - y.{i})) > eps then raise DoesNotMatch
    done ;
    true
  with DoesNotMatch -> false

let distance x y =
  let sum = ref 0. in
  for i = 0 to Bigarray.Array1.dim x - 1 do
    sum := !sum +. (((float_of_int (x.{i} - y.{i})) /. 256.) ** 2.)
  done ;
  50. +. (50. *. !sum)
