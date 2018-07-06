open Hashtbl
open Random
open Printf
open Plotter
open Utils2
open Renderer

type var =    Name of string
            | Unit
            (*| Indefinite*)
            | Double of var | Half of var
            | Next of var | Prev of var
            | Opposite of var
            | Divide of var * var

type innerValues = var option
                 * var option
                 * var option
                 * var option

type shapeprogram = Concat of shapeprogram * shapeprogram
             | Embed of shapeprogram
             | Turn of var option
             | Repeat of var option * shapeprogram
             | Integrate of var option * bool option * innerValues
             | Define of string * var
             | Nop

type internal = { mutable x             : float
                ; mutable y             : float
                ; mutable face          : float
                ; mutable speed         : float
                ; mutable accel         : float
                ; mutable angularSpeed  : float
                ; mutable angularAccel  : float }

exception MalformedProgram of string

let rec my_print_var v = match v with
    | Name s -> s
    (*| Indefinite -> "indefinite"*)
    | Unit -> "unit"
    | Double v' -> "Double(" ^ (my_print_var v') ^ ")"
    | Half v' -> "Half(" ^ (my_print_var v') ^ ")"
    | Next v' -> "Next(" ^ (my_print_var v') ^ ")"
    | Prev v' -> "Prev(" ^ (my_print_var v') ^ ")"
    | Opposite v' -> "Opposite(" ^ (my_print_var v') ^")"
    | Divide (v1,v2) ->
        "Divide(" ^ (my_print_var v1) ^ ","^(my_print_var v2)^")"

let (++) pr1 pr2 = Concat(pr1, pr2)

let scale = d_from_origin /. 4.5
let steps = 20.
let isteps = int_of_float steps
let ratio = scale /. steps

let replace_stroke curr future =
    curr.face <- future.face ;
    curr.speed <- future.speed ;
    curr.accel <- future.accel ;
    curr.angularSpeed <- future.angularSpeed ;
    curr.angularAccel <- future.angularAccel

let empty_state =
        { x = -1.
        ; y = -1.
        ; face = 0.
        ; speed = 1.
        ; accel = 0.
        ; angularSpeed = 0.
        ; angularAccel = 0. }


let replace curr future =
    curr.x <- future.x ;
    curr.y <- future.y ;
    curr.face <- future.face ;
    curr.speed <- future.speed ;
    curr.accel <- future.accel ;
    curr.angularSpeed <- future.angularSpeed ;
    curr.angularAccel <- future.angularAccel

let (<<-) curr future = replace curr future

let pp_shapeprogram shapeprogram =
    let rec pp_helper shapeprogram tabs = match shapeprogram with
        | Concat (p1,p2) ->
            let s1 = pp_helper p1 tabs in
            let s2 = pp_helper p2 tabs in
            Printf.sprintf "%s ;\n%s" s1 s2
        | Turn f -> Printf.sprintf "%sTurn%s" tabs
            (match f with
            | None -> ""
            | Some(f) -> sprintf "(angle=%s)" (my_print_var f))
        | Embed pr ->
            let s = pp_helper pr (tabs^"  ") in
            sprintf "%sEmbed {\n%s\n%s}" tabs s tabs
        | Repeat (n,pr) ->
            let s = pp_helper pr (tabs^"  ") in
            sprintf "%sRepeat%s {\n%s\n%s}"
                tabs
                (match n with
                | None -> ""
                | Some(m) when m = Unit -> ""
                | Some(m) -> "("^my_print_var m^")")
                s tabs
        | Integrate (f,pen,(speed,accel,angularSpeed,angularAccel)) ->
            sprintf "%sIntegrate%s"
            tabs
            (if f = None && pen = None && speed = None && accel = None &&
                angularSpeed = None && angularAccel = None then ""
            else (let s = ref "(" in begin
                (match f with None->()|Some(f)->s:=!s^"t="^my_print_var f^",");
                (match pen with None->()|Some(p)->s:=!s^"pen="^(my_print_bool p)^",");
                (match speed with None->()|Some(f)->s:=!s^"speed="^my_print_var f^",");
                (match accel with None->()|Some(f)->s:=!s^"accel="^my_print_var f^",");
                (match angularSpeed with None->()|Some(f)->s:=!s^"angularSpeed="^my_print_var f^",");
                (match angularAccel with None->()|Some(f)->s:=!s^"angularAccel="^my_print_var f^",");
            end ; ((String.sub !s 0 ((String.length !s) - 1))^")")))
        | Define (name,v) ->
            sprintf "%s%s = %s" tabs name (my_print_var v)
        | Nop ->
            sprintf "nop"
    in
    pp_helper shapeprogram ""

let valuesCostVar : var -> int =
    fun v -> match v with
    | Unit -> 1
    (*| Indefinite -> 1*)
    | Name _ -> 1
    | Double v' ->  1
    | Half v' ->  1
    | Next v' ->  1
    | Prev v' ->  1
    | Opposite v' ->  1
    | Divide(v1,v2) -> 1

let costVar : var option -> int =
    let rec helper v = match v with
        | Unit -> valuesCostVar Unit
        (*| Indefinite -> valuesCostVar Indefinite*)
        | Name s -> valuesCostVar (Name s)
        | Double v' ->  (valuesCostVar (Double v')) + helper v'
        | Half v' ->  (valuesCostVar (Half v')) + helper v'
        | Prev v' ->  (valuesCostVar (Prev v')) + helper v'
        | Next v' ->  (valuesCostVar (Next v')) + helper v'
        | Opposite v' ->  (valuesCostVar (Opposite v')) + helper v'
        | Divide(v1,v2) -> (valuesCostVar (Divide(v1,v2))) + helper v1 + helper v2
    in fun vo -> match vo with
    | None -> 0
    | Some v -> helper v

let valuesCostProgram : shapeprogram -> int =
    fun p -> match p with
    | Turn _ -> 1
    | Embed _ -> 1
    | Concat (_,_) -> 1
    | Repeat (_,_) -> 1
    | Define (_,_) -> 1
    | Nop -> 0
    | Integrate(_,_,_) -> 1

let evaluateVar v htbl_var =
    let rec evaluateVar_helper v htbl_var = match v with
    | Unit -> 1.
    (*| Indefinite -> 10. +. (float_of_int (Random.int 5))*)
    | Double v' -> 2.*.(evaluateVar_helper v' htbl_var)
    | Half v' -> (evaluateVar_helper v' htbl_var) /. 2.
    | Prev v' -> (evaluateVar_helper v' htbl_var) -. 1.
    | Next v' -> (evaluateVar_helper v' htbl_var) +. 1.
    | Opposite v' -> (-1.)*.(evaluateVar_helper v' htbl_var)
    | Divide (v1,v2) ->
        (evaluateVar_helper v1 htbl_var) /. (evaluateVar_helper v2 htbl_var)
    | Name s ->
        if Hashtbl.mem htbl_var s then
            let value = Hashtbl.find htbl_var s in
            Hashtbl.remove htbl_var s ;
            let v = evaluateVar_helper value htbl_var in
            Hashtbl.add htbl_var s value ;
            v
        else evaluateVar_helper Unit htbl_var
    in match evaluateVar_helper v htbl_var with
    | n when n = nan -> raise (MalformedProgram("Some var was NaN"))
    | f -> f

let rec costProgram : shapeprogram -> int =
    fun p -> match p with
    | Turn v -> (valuesCostProgram (Turn v)) + (costVar v)
    | Embed (p) -> (valuesCostProgram (Embed(p))) + costProgram p
    | Concat (p1,p2) -> (valuesCostProgram (Concat (p1,p2)))
                        + costProgram p1 + costProgram p2
    | Repeat (v,p') -> (valuesCostProgram (Repeat(v,p')))
                       + (costVar v) + (costProgram p')
    | Nop -> 0
    | Define (s,v) -> (valuesCostProgram (Define (s,v)))
                      + costVar (Some v)
    | Integrate(v1,v2,(v3,v4,v5,v6)) ->
            (valuesCostProgram (Integrate(v1,v2,(v3,v4,v5,v6))))
          + costVar v1
          + (match v2 with None -> 0 | Some _ -> 1)
          + costVar v3
          + costVar v4
          + costVar v5
          + costVar v6

let rec costProgram_norepeat : shapeprogram -> int =
    fun p -> match p with
    | Turn v -> (valuesCostProgram (Turn v)) + (costVar v)
    | Embed (p) -> (valuesCostProgram (Embed(p))) + costProgram_norepeat p
    | Concat (p1,p2) -> (valuesCostProgram (Concat (p1,p2)))
                        + costProgram_norepeat p1 + costProgram_norepeat p2
    | Repeat (v,p') -> ((valuesCostProgram (Repeat(v,p')))
                         + (costVar v) + (costProgram_norepeat p'))
                       * (match v with
                            | None -> 2
                            | Some(v') -> (int_of_float (evaluateVar v'
                            (Hashtbl.create 101))))
    | Nop -> 0
    | Define (s,v) -> (valuesCostProgram (Define (s,v)))
                      + costVar (Some v)
    | Integrate(v1,v2,(v3,v4,v5,v6)) ->
            (valuesCostProgram (Integrate(v1,v2,(v3,v4,v5,v6))))
          + costVar v1
          + (match v2 with None -> 0 | Some _ -> 1)
          + costVar v3
          + costVar v4
          + costVar v5
          + costVar v6

let rec costProgram_noembed : shapeprogram -> int =
    fun p -> match p with
    | Turn v -> (valuesCostProgram (Turn v)) + (costVar v)
    | Embed (p) -> 2 * ((valuesCostProgram (Embed(p))) + costProgram_noembed p)
    | Concat (p1,p2) -> (valuesCostProgram (Concat (p1,p2)))
                        + costProgram_noembed p1 + costProgram_noembed p2
    | Repeat (v,p') -> (valuesCostProgram (Repeat(v,p')))
                       + (costVar v) + (costProgram_noembed p')
    | Nop -> 0
    | Define (s,v) -> (valuesCostProgram (Define (s,v)))
                      + costVar (Some v)
    | Integrate(v1,v2,(v3,v4,v5,v6)) ->
            (valuesCostProgram (Integrate(v1,v2,(v3,v4,v5,v6))))
          + costVar v1
          + (match v2 with None -> 0 | Some _ -> 1)
          + costVar v3
          + costVar v4
          + costVar v5
          + costVar v6

let interpret ?factor:(factor=1.) ?noise:(noise=false) shapeprogram =
    let r_canvas = ref (new_canvas ()) in
    let moveto x y = (r_canvas := moveto !r_canvas x y)
    and lineto x y = (r_canvas := lineto !r_canvas x y) in
    let rec inter ?sizes:(sizes=None) shapeprogram htbl_var curr_state =
        match shapeprogram with
        | Embed p ->
            let save_state =
                {curr_state with x = curr_state.x} in
            let htbl_var' = Hashtbl.copy htbl_var in
            inter ~sizes p htbl_var' curr_state ;
            replace curr_state save_state ;
            moveto curr_state.x curr_state.y
        | Turn f ->
                let angle : float = match f with
                  | None -> 1.
                  | Some(f') -> evaluateVar f' htbl_var
                in
                curr_state.face <-
                  curr_state.face +.
                  (angle *. pis2) *.
                    (1. +. if noise then (normal_random ()) *. 0.05 else 0.)
        | Concat (p1,p2) ->
            inter ~sizes p1 htbl_var curr_state ;
            inter ~sizes p2 htbl_var curr_state
        | Repeat (n, pr) ->
            let n' = int_of_float (match n with
                | None -> 2.
                | Some v -> evaluateVar v htbl_var) in
            for i = 1 to n' do
                inter ~sizes pr htbl_var curr_state
            done
        | Integrate (f, pen, (speed,accel,angularSpeed,angularAccel)) ->
            let f = match f with None -> 1.
                    | Some v -> evaluateVar v htbl_var in
            let speed =
                (match speed with None -> 1.
                    | Some v -> evaluateVar v htbl_var) and
            accel =
                (match accel with None -> 0.
                    | Some v -> evaluateVar v htbl_var) and
            angularSpeed =
                (match angularSpeed with None -> 0.
                    | Some v -> evaluateVar v htbl_var) and
            angularAccel =
                (match angularAccel with None -> 0.
                    | Some v -> evaluateVar v htbl_var) in
            curr_state.speed <- speed ;
            curr_state.accel <- accel ;
            curr_state.angularSpeed <- angularSpeed ;
            curr_state.angularAccel <- angularAccel ;
            let pen = match pen with | None -> true | Some b -> b in
            for i = 1 to (int_of_float (f *. steps /. factor)) do
                let futur_x =
                  curr_state.x
                  +. (curr_state.speed *. cos(curr_state.face))
                     *. ratio
                  +. if noise then 0.001 *. (normal_random ()) else 0.
                and futur_y =
                  curr_state.y
                  +. (curr_state.speed *. sin(curr_state.face))
                     *. ratio
                  +. if noise then 0.001 *. (normal_random ()) else 0. in
                if pen then lineto futur_x futur_y
                       else moveto futur_x futur_y ;
                curr_state.x <- futur_x ;
                curr_state.y <- futur_y ;
                curr_state.face <-
                  curr_state.face +.
                  (pi2 *. curr_state.angularSpeed *. factor ) /. (steps) +.
                  if noise then 0.01 *. (normal_random ()) else 0.;
                curr_state.speed <-
                  curr_state.speed +.
                  (curr_state.accel *. factor /. (steps)) +.
                  if noise then 0.01 *. (normal_random ()) else 0.;
                curr_state.angularSpeed <-
                    curr_state.angularSpeed (* KINDA BROKEN *)
                    +. (2. *. curr_state.angularAccel /. (steps)) ;
            done
        | Define (name,v) -> Hashtbl.add htbl_var name v
        | Nop -> ()
    in
    let initial_state =
      { x = middle_x !r_canvas
      ; y = middle_y !r_canvas
      ; face = (if noise then (0.1 -. Random.float 0.2) else 0.)
      ; speed = 1.
      ; accel = 0.
      ; angularSpeed = 0.
      ; angularAccel = 0. } in
    inter shapeprogram (Hashtbl.create 101) initial_state ;
    let canvas = !r_canvas in
    r_canvas := (new_canvas ());
    canvas

let interpret_normal ?noise:(noise=false) p =
  let rec remove_start found p = match p with
    | Nop -> Nop, found
    | Define (_,_) -> p,found
    | Turn f -> (if found then Turn f else Nop),found
    | Integrate(a,Some(false),c) ->
        if found then (p,found)
        else Nop,false
    | Integrate(a,pen,c) -> p,true
    | Concat(p1,p2) ->
        if found then p,found
        else begin
          let (p',f') = remove_start found p1 in
          if f' then Concat(p',p2),f'
          else begin
            let (p'',f'') = remove_start f' p2 in
            Concat(p',p''),f''
          end
        end
    | Embed p' ->
        if found then (p,found)
        else begin
          let (p'',f) = remove_start found p' in
          if f then Embed(p''),f
          else Nop,false
        end
    | Repeat(v,p') -> p,found
  in
  let rec find_bit p = match p with
    | Turn(Some(v)) ->
        let f = evaluateVar v (Hashtbl.create 101) in
        (true,(f > 0.))
    | Turn(None) -> (true,true)
    | Concat(p1,p2) ->
        let (b1,b2) = find_bit p1 in
        if b1 then b1,b2 else (find_bit p2)
    | Embed (p) -> false,false
    | Repeat (v,p) -> find_bit p
    | _ -> false,false
  in
  let rec swap_bit p = match p with
    | Turn(Some(v)) -> Turn(Some(Opposite(v)))
    | Concat(p1,p2) -> Concat(swap_bit p1, swap_bit p2)
    | Embed(p) -> Embed(swap_bit p)
    | Repeat(v,p) -> Repeat(v, swap_bit p)
    | x -> x
  in
  let rec min_t p = match p with
    | Integrate(Some(t),_,_) -> evaluateVar t (Hashtbl.create 101)
    | Integrate(None,_,_) -> 1.
    | Concat(p1,p2) -> min (min_t p1) (min_t p2)
    | Embed(p) -> min_t p
    | Repeat(_,p) -> min_t p
    | _ -> 99999.
  in
  let p',_ = remove_start false p in
  let _,b = find_bit p' in
  let p'' = if (not b) then (swap_bit p') else p' in
  let factor = min_t p'' in
  (*print_endline (pp_shapeprogram p) ;*)
  (*print_newline () ;*)
  (*print_endline (pp_shapeprogram p'') ;*)
  (*print_newline () ;*)
  (*print_float factor ;*)
  (*print_newline () ;*)
  (*print_endline "=========================" ;*)
  interpret ~factor ~noise p''
