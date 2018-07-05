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

let logo_NOP : turtle = fun s -> ([], s)

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
  let l,_ = (t logo_NOP) (init_state ()) in
  List.iter
    (fun e ->
       pp_logo_instruction e ;
       prerr_string "; " ;
       flush_everything () )
    l ;
  prerr_newline ()

let eval_normal_turtle t2t =
  let p,_ = (t2t logo_NOP) (init_state ()) in
  let c = ref (new_canvas ()) in
  let lineto x y = (c := (lineto !c x y))
  and moveto x y = (c := (moveto !c x y)) in
  let t = init_state () in
  moveto t.x t.y ;
  let rec eval_instruction n i = match i with
    | PU         -> t.p <- false
    | PD         -> t.p <- true
    | RT(angle)  -> t.t <- t.t +. angle
    | SET(state) -> t <<- state ; moveto (t.x) (t.y) ;
    | FW(length) ->
        let x = t.x +. (length *. cos(t.t))
        and y = t.y +. (length *. sin(t.t)) in
        (if t.p then lineto else moveto) x y ;
        t.x <- x ;
        t.y <- y
  in
  let rec remove_start pd l = match l with
    | [] -> []
    | PU :: r -> remove_start false r
    | PD :: r -> remove_start true r
    | RT(_)::r -> remove_start pd r
    | FW(0.)::r -> remove_start pd r
    | FW(_)::r -> if pd then l else remove_start pd r
    | SET(_)::r -> remove_start pd r
  in
  let rec scale l f = match l with
    | [] -> []
    | FW(x)::r when (abs_float x < 0.2) -> FW(x)::(scale r f)
    | FW(x)::r -> FW(x*.f)::(scale r f)
    | e::r -> e::(scale r f)
  in
  let rec find_min l = match l with
    | [] -> 99999.
    | FW(x)::r when (abs_float x) < 0.2 -> find_min r
    | FW(x)::r -> min (x) (find_min r)
    | e::r -> find_min r
  in
  let rec mirror found bit l = match l with
    | [] -> []
    | RT(x) :: r ->
        if not found then RT(abs_float(x)) :: (mirror true (x > 0.) r)
        else RT(if bit then x else (-. x)) :: (mirror found bit r)
    | x :: r -> x :: (mirror found bit r)
  in
  let p' = remove_start true p in
  let m = find_min p' in
  let p'' = scale p' (1./.m) in
  let p''' = mirror false true p'' in
  List.iteri eval_instruction p''' ;
  !c

let eval_turtle ?sequence t2t =
  let p,_ = (t2t logo_NOP) (init_state ()) in
  let c = ref (new_canvas ()) in
  let lineto x y = (c := (lineto !c x y))
  and moveto x y = (c := (moveto !c x y)) in
  let t = init_state () in
  moveto t.x t.y ;
  let rec eval_instruction n i = match i with
    | PU         -> t.p <- false
    | PD         -> t.p <- true
    | RT(angle)  -> t.t <- t.t +. angle
    | SET(state) ->
        begin
          t <<- state ;
          moveto (t.x) (t.y) ;
          match sequence with
            | None -> ()
            | Some path ->
                let l_c = ref (new_canvas ()) in
                l_c := circle !l_c t.x t.y ;
                let name = Printf.sprintf "%s%03d.png" path n in
                output_canvas_png !l_c 512 name
        end
    | FW(length) ->
        let x = t.x +. (length *. cos(t.t))
        and y = t.y +. (length *. sin(t.t)) in
        (if t.p then lineto else moveto) x y ;
        t.x <- x ;
        t.y <- y ;
        match sequence with
          | None -> ()
          | Some path ->
              let l_c = ref (new_canvas ()) in
              l_c := circle !l_c x y ;
              let name = Printf.sprintf "%s%03d.png" path n in
              output_canvas_png !l_c 512 name
  in
  List.iteri eval_instruction p ;
  !c

let logo_PU : turtle =
  fun s -> ([PU], {s with p = false})

let logo_PD : turtle =
  fun s -> ([PD], {s with p = true})

let logo_RT : float -> turtle =
  fun angle -> fun s -> ([RT(angle *. 4. *. atan(1.))], {s with t = s.t +. angle})

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

let logo_SET : (state -> turtle) = fun s -> fun _ -> ([SET({s with t=s.t *. 4. *. atan(1.)})], s)

(*let logo_CHEAT : float -> turtle =*)
  (*fun length ->*)
    (*(logo_SEQ (logo_FW length) (logo_RT (logo_var_HLF (logo_var_HLF logo_var_UNIT))))*)


(*let logo_CHEAT2 : float -> turtle =*)
  (*fun length ->*)
    (*(logo_SEQ (logo_FW length) (logo_RT (logo_var_HLF (logo_var_HLF (logo_var_HLF logo_var_UNIT)))))*)

(*let logo_CHEAT3 : float -> turtle =*)
  (*fun length ->*)
    (*(logo_SEQ (logo_FW length) (logo_RT (logo_var_HLF (logo_var_HLF (logo_var_HLF (logo_var_HLF logo_var_UNIT))))))*)

(*let logo_CHEAT4 : float -> turtle =*)
  (*fun length ->*)
    (*(logo_SEQ (logo_FW length) (logo_RT (logo_var_HLF logo_var_UNIT)))*)

let turtle_to_list turtle =
  let l,_ = (turtle logo_NOP) (init_state ()) in l

let turtle_to_png turtle resolution filename =
  output_canvas_png (eval_turtle turtle) resolution filename

let turtle_to_array turtle resolution =
  canvas_to_1Darray (eval_turtle turtle) resolution

let normal_turtle_to_array turtle resolution =
  canvas_to_1Darray (eval_normal_turtle turtle) resolution

let turtle_to_both turtle resolution filename =
  let c = (eval_turtle turtle) in
  output_canvas_png c resolution filename ;
  canvas_to_1Darray c resolution

let max = 28. *. 256.

let distance x y =
  let sum = ref 0. in
  for i = 0 to Bigarray.Array1.dim x - 1 do
    sum := !sum +. (((float_of_int (x.{i} - y.{i})) /. 256.) ** 2.)
  done ;
  50. +. (50. *. !sum)
