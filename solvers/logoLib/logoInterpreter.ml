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

let init_state () = {x = d_from_origin; y = d_from_origin; t = 0.; p = true}

let flush_everything () =
  Pervasives.flush stdout;
  Pervasives.flush stderr

let pp_logo_instruction i =
  match i with
  | PU     -> prerr_string "PU"
  | PD     -> prerr_string "PD"
  | FW(a)  -> prerr_string "FW(" ; prerr_float a ; prerr_string ")"
  | RT(a)  -> prerr_string "RT(" ; prerr_float a ; prerr_string ")"
  | SET(s) ->
      prerr_string "SET(" ;
      prerr_float s.x ;
      prerr_string "," ;
      prerr_float s.y ;
      prerr_string "," ;
      prerr_float s.t ;
      prerr_string "," ;
      prerr_string (if s.p then "on" else "off") ;
      prerr_string ")"

let pp_turtle t =
  let l,_ = t (init_state ()) in
  List.iter
    (fun e ->
       pp_logo_instruction e ;
       prerr_string "; " ;
       flush_everything () )
    l ;
  prerr_newline ()

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

let logo_var_UNIT : float  = 1.
let logo_var_TWO : float   = 2.
let logo_var_THREE: float  = 3.
let logo_var_PI   : float  = 3.14159265359
let logo_var_NXT         f = f +. 1.
let logo_var_PRV         f = f -. 1.
let logo_var_DBL         f = f *. 2.
let logo_var_HLF         f = f /. 2.
let logo_var_ADD      f f' = f +. f'
let logo_var_SUB      f f' = f -. f'
let logo_var_MUL      f f' = f *. f'
let logo_var_DIV      f f' = f /. f'

let turtle_to_png turtle resolution filename =
  output_canvas_png (eval_turtle turtle) resolution filename

let turtle_to_array turtle resolution =
  canvas_to_1Darray (eval_turtle turtle) resolution

let turtle_to_both turtle resolution filename =
  let c = (eval_turtle turtle) in
  output_canvas_png c resolution filename ;
  canvas_to_1Darray c resolution
