open VGWrapper

type state =
  { mutable x : float
  ; mutable y : float
  ; mutable t : float
  ; mutable p : bool
  }

type logo_instruction =
  | PU          | PD
  | FW of float | RT of float
  | SET of state

type program = logo_instruction list

type turtle = (state -> (program * state))

let (<<-) t1 t2 =
  t1.x <- t2.x ;
  t1.y <- t2.y ;
  t1.t <- t2.t ;
  t1.p <- t2.p

let init_state () = {x = 200.; y = 200.; t = 0.; p = true}

let pp_logo_instruction i = match i with
  | PU     -> print_string "PU"
  | PD     -> print_string "PD"
  | FW(a)  -> print_string "FW(" ; print_float a ; print_string ")"
  | RT(a)  -> print_string "RT(" ; print_float a ; print_string ")"
  | SET(s) ->
      print_string "SET(" ;
      print_float s.x ;
      print_string "," ;
      print_float s.y ;
      print_string "," ;
      print_float s.t ;
      print_string "," ;
      print_string (if s.p then "on" else "off") ;
      print_string ")"

let pp_turtle t =
  let l,_ = t (init_state ()) in
  List.iter (pp_logo_instruction) l ;
  print_newline ()

let eval_turtle turtle =
  let p,_ = turtle (init_state ()) in
  let c = ref (new_canvas ()) in
  let lineto x y = (c := (lineto !c x y))
  and moveto x y = (c := (moveto !c x y)) in
  let t = init_state () in
  moveto t.x t.y ;
  let rec eval_instruction i = match i with
    | PU         -> t.p <- false
    | PD         -> t.p <- true
    | RT(angle)  -> t.t <- t.t +. angle
    | SET(state) -> t <<- state ; moveto (t.x) (t.y)
    | FW(length) ->
        let x = t.x +. (length *. cos(t.t))
        and y = t.y +. (length *. sin(t.t)) in
        (if t.p then lineto else moveto) x y ;
        t.x <- x ;
        t.y <- y
  in
  List.iter pp_logo_instruction p ;
  List.iter eval_instruction p ;
  !c

let logo_PU : turtle =
  fun s -> ([PU], {s with p = false})

let logo_PD : turtle =
  fun s -> ([PD], {s with p = true})

let logo_RT : float -> turtle =
  fun angle -> fun s -> ([RT(angle)], {s with t = s.t +. angle})

let logo_FW : float -> turtle =
  fun length  ->
    fun s -> ([FW(length)], {s with
                               x = s.x +. (length *. cos(s.t)) ;
                               y = s.y +. (length *. sin(s.t)) })

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

let logo_SET : (state -> turtle) = fun s -> fun _ -> ([SET(s)], s)

let logo_NOP : turtle = fun s -> ([], s)

let logo_var_UNIT : float  = 100.
let logo_var_NEXT        f = f +. logo_var_UNIT

let turtle_to_png turtle resolution filename =
  output_canvas_png (eval_turtle turtle) resolution filename

let turtle_to_array turtle resolution =
  canvas_to_1Darray (eval_turtle turtle) resolution

let turtle_to_both turtle resolution filename =
  let c = (eval_turtle turtle) in
  output_canvas_png c resolution filename ;
  canvas_to_1Darray c resolution
