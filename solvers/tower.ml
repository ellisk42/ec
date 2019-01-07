open Core

open Client
open Timeout
open Task
open Utils
open Program
open Type

type tower_state = {hand_position : int;
                    hand_orientation : int;}
let empty_tower_state =
  {hand_position = 0;
   hand_orientation = 1;}
                   
(* ttower = state -> (state, list of blocks) *)
type tt = tower_state -> tower_state * ( (int*int*int) list)

let tower_sequence (a : tt) (b : tt) : tt = fun hand ->
  let hand, a' = a hand in
  let hand, b' = b hand in
  (hand, a' @ b');;
let empty_tower : tt = fun h -> (h,[]);;


let tower_extent p =
  let xs = p |> List.map ~f:(fun (x,_,_) -> x) in
  let x1 = List.fold_left ~init:(List.hd_exn xs) ~f:max xs in
  let x0 = List.fold_left ~init:(List.hd_exn xs) ~f:min xs in
  x1 - x0

let center_tower p =
  let xs = p |> List.map ~f:(fun (x,_,_) -> x) in
  let x1 = List.fold_left ~init:(List.hd_exn xs) ~f:max xs in
  let x0 = List.fold_left ~init:(List.hd_exn xs) ~f:min xs in
  (* bounding box: [x0,x1] *)
  let c = (x1-x0)/2 + x0 in
  p |> List.map ~f:(fun (x,w,h) -> (x-c,w,h))


let block w h =
  let n = Printf.sprintf "%dx%d" w h in
  let xOffset = 0 in
  let w = 2*w in
  let h = 2*h in
  let v : tt -> tt = fun k : tt ->
    fun hand ->
      let (hand', rest) = k hand in
      (hand', (xOffset + hand.hand_position, w, h) :: rest)
  in
  ignore(primitive n (ttower @> ttower) v)
;;

block 3 1;;
block 1 3;;
block 1 1;;
block 2 1;;
block 1 2;;
block 4 1;;
block 1 4;;

ignore(primitive "left" (tint @> ttower @> ttower)
         (let f : int -> tt -> tt = fun (d : int) ->
             fun (k : tt) ->
             fun (hand : tower_state) ->
               let hand' = {hand with hand_position = hand.hand_position - d} in
               let (hand'', rest) = k hand' in
               (hand'', rest)
          in f));;
ignore(primitive "right" (tint @> ttower @> ttower)
         (let f : int -> tt -> tt = fun (d : int) ->
             fun (k : tt) ->
             fun (hand : tower_state) ->
               let hand' = {hand with hand_position = hand.hand_position + d} in
               let (hand'', rest) = k hand' in
               (hand'', rest)
          in f));;
ignore(primitive "tower_loop" (tint @> (tint @> ttower) @> ttower @> ttower)
         (let rec f (start : int) (stop : int) (body : int -> tt) : tt = fun (hand : tower_state) -> 
             if start >= stop then (hand,[]) else
               let (hand', thisIteration) = body start hand in
               let (hand'', laterIterations) = f (start+1) stop body hand' in
               (hand'', thisIteration @ laterIterations)
          in fun (n : int) (b : int -> tt) (k : tt) : tt -> fun (hand : tower_state) -> 
            let (hand, body_blocks) = f 0 n b hand in
            let hand, later_blocks = k hand in
            (hand, body_blocks @ later_blocks)));;
ignore(primitive "tower_loopM" (tint @> (tint @> ttower @> ttower) @> ttower @> ttower)
         (fun i (f : int -> tt -> tt) (z : tt) : tt -> List.fold_right (0 -- (i-1)) ~f ~init:z));;
ignore(primitive "tower_embed" ((ttower @> ttower) @> ttower @> ttower)
         (fun (body : tt -> tt) (k : tt) : tt ->
            fun (hand : tower_state) ->
              let (_, bodyActions) = body empty_tower hand in
              let (hand', laterActions) = k hand in
              (hand', bodyActions @ laterActions)));;
ignore(primitive "moveHand" (tint @> ttower @> ttower)
         (fun (d : int) (k : tt) : tt ->
            fun (state : tower_state) ->
              k {state with hand_position = state.hand_position + state.hand_orientation*d}));;
ignore(primitive "reverseHand" (ttower @> ttower)
         (fun (k : tt) : tt ->
            fun (state : tower_state) ->
              k {state with hand_orientation = -1*state.hand_orientation}));;
            

let simulate_without_physics plan =
  let overlaps (x,w,h) (x',y',w',h')  =
    let x1 = x - w/2 in
    let x2 = x + w/2 in
    let x1' = x' - w'/2 in
    let x2' = x' + w'/2 in
    if x1' >= x2 || x1 >= x2' then None else
      Some(y' + h/2 + h'/2)
  in

  let lowest_possible_height (_,_,h) = h/2 in
  let place_at_height (x,w,h) y = (x,y,w,h) in 

  let place_block world block =
    let lowest = List.filter_map world ~f:(overlaps block) |>
                 List.fold_right ~init:(lowest_possible_height block) ~f:max
    in
    place_at_height block lowest :: world
  in

  let rec run plan world = match plan with
    | [] -> world
    | b :: bs -> run bs (place_block world b)
  in
  let simulated = run plan [] |> List.sort ~compare:(fun x y ->
      if x > y then 1 else if x < y then -1 else 0
    ) in
  simulated
;;

let blocks_extent blocks =
  if blocks = [] then 0 else
  let xs = blocks |> List.map ~f:(fun (x,_,_,_) -> x) in
  let x1 = List.fold_left ~init:(List.hd_exn xs) ~f:max xs in
  let x0 = List.fold_left ~init:(List.hd_exn xs) ~f:min xs in
  x1 - x0

let tower_height blocks =
  if blocks = [] then 0 else
    let ys = blocks |> List.map ~f:(fun (_,y,_,h) -> y + h/2) in
    let y1 = List.fold_left ~init:(List.hd_exn ys) ~f:max ys in    
    let ys = blocks |> List.map ~f:(fun (_,y,_,h) -> y - h/2) in
    let y0 = List.fold_left ~init:(List.hd_exn ys) ~f:min ys in
    y1 - y0

let show_tower dt =
  dt |> List.map ~f:(fun (a,b,c,d) ->
      Printf.sprintf "BLOCK{w=%d, h=%d, x=%d, y=%d}"
        c d a b) |> join ~separator:"\n"

let recent_tower : (program option) ref = ref None;;
let recent_discrete : ((int*int*int*int) list) ref = ref [];;

let evaluate_discrete_tower_program timeout p =
  match !recent_tower with
  | Some(p') when program_equal p p' -> !recent_discrete
  | _ ->
    begin
      recent_tower := Some(p);
      (* Printf.eprintf "%s\n" (string_of_program p); *)
      let p = analyze_lazy_evaluation p in
      let new_discrete = 
        try
          match run_for_interval
                  timeout
                  (fun () -> run_lazy_analyzed_with_arguments p [fun s -> (s, [])] empty_tower_state |> snd)
          with
          | Some(p) ->
            let p = center_tower p in
            let t = simulate_without_physics p in
            t
            (* (Printf.eprintf "%s\n" (show_tower t); t) *)
          | _ -> []
        with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
             (* we have to be a bit careful with exceptions *)
             (* if the synthesized program generated an exception, then we just terminate w/ false *)
             (* but if the enumeration timeout was triggered during program evaluation, we need to pass the exception on *)
             | otherException -> begin
                 if otherException = EnumerationTimeout then raise EnumerationTimeout else []
               end
      in
      recent_discrete := new_discrete;
      new_discrete
    end
;;

register_special_task "supervisedTower" (fun extra ?timeout:(timeout = 0.001)
    name task_type examples -> 
  assert (task_type = ttower @> ttower);
  assert (examples = []);

  let open Yojson.Basic.Util in
  
  let plan = extra |> member "plan" |> to_list |> List.map ~f:(fun command ->
      match command |> to_list with
      | [a;b;c;] -> (a |> to_int, b |> to_int, c |> to_int)
      |_ -> assert false) |> center_tower |> simulate_without_physics
  in
  (* Printf.eprintf "TARGETING:\n%s\n\n" *)
  (*   (show_tower plan); *)

  { name = name    ;
    task_type = task_type ;
    log_likelihood =
      (fun p ->
         let hit = evaluate_discrete_tower_program timeout p = plan in
         (* Printf.eprintf "\t%b\n\n" hit; *)
         if hit then 0. else log 0.)
  })
;;

