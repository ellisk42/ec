open Core

open Client
open Timeout
open Task
open Utils
open Program
open Type

(* ttower = state -> (state, list of blocks) *)
type tt = int -> int * ( (int*int*int) list)

let tower_sequence (a : tt) (b : tt) : tt = fun hand ->
  let hand, a' = a hand in
  let hand, b' = b hand in
  (hand, a' @ b');;
let empty_tower : tt = fun h -> (h,[]);;

let ttower = make_ground "tower";;

let maximum_number_of_blocks = 35;;
let maximum_tower_extent = 15.;;

type tower_result = {stability : float;
                     length : float;
                     area : float;
                     staircase : float;
                     height : float;
                     overpass : float;}

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

(* let discrete_plan p = *)
(*   p |> List.map ~f:(fun (x,w,h) -> *)
(*       (round (x*.10.) |> Int.of_float, *)
(*        round (w*.10.) |> Int.of_float, *)
(*        round (h*.10.) |> Int.of_float))c *)

let block w h =
  let n = Printf.sprintf "%dx%d" w h in
  let xOffset = if w mod 2 = 1 then 5 else 0 in
  let e = 1 in
  (* This will ensure that the width is an even number *)
  let w = 10*w - 2 * e in
  let h = 10*h in
  let v : tt -> tt = fun k : tt ->
    fun hand ->
      let (hand', rest) = k hand in
      (hand', (xOffset + hand, w, h) :: rest)
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
             fun (hand : int) ->
               let hand' = hand - 5*d in
               let (hand'', rest) = k hand' in
               (hand'', rest)
          in f));;
ignore(primitive "right" (tint @> ttower @> ttower)
         (let f : int -> tt -> tt = fun (d : int) ->
             fun (k : tt) ->
             fun (hand : int) ->
               let hand' = hand + 5*d in
               let (hand'', rest) = k hand' in
               (hand'', rest)
          in f));;
ignore(primitive "tower_loop" (tint @> (tint @> ttower) @> ttower @> ttower)
         (let rec f (start : int) (stop : int) (body : int -> tt) : tt = fun (hand : int) -> 
             if start >= stop then (hand,[]) else
               let (hand', thisIteration) = body start hand in
               let (hand'', laterIterations) = f (start+1) stop body hand' in
               (hand'', thisIteration @ laterIterations)
          in fun (n : int) (b : int -> tt) (k : tt) : tt -> fun (hand : int) -> 
            let (hand, body_blocks) = f 0 n b hand in
            let hand, later_blocks = k hand in
            (hand, body_blocks @ later_blocks)));;
ignore(primitive "tower_loopM" (tint @> (tint @> ttower @> ttower) @> ttower @> ttower)
         (fun i (f : int -> tt -> tt) (z : tt) : tt -> List.fold_right (0 -- (i-1)) ~f ~init:z));;
ignore(primitive "tower_embed" ((ttower @> ttower) @> ttower @> ttower)
         (fun (body : tt -> tt) (k : tt) : tt ->
            fun (hand : int) ->
              let (_, bodyActions) = body empty_tower hand in
              let (hand', laterActions) = k hand in
              (hand', bodyActions @ laterActions)));;             
            
let tower_socket = connect_socket 9494;;

let send_to_tower_server k =
  socket_json tower_socket k
    
let parse_tower_result result =
  let open Yojson.Safe.Util in
  let stability = result |> member "stability" |> to_float in
  let length = result |> member "length" |> to_float in
  let area = result |> member "area" |> to_float in
  let staircase = result |> member "staircase" |> to_float in
  let height = result |> member "height" |> to_float in
  let overpass = result |> member "overpass" |> to_float in
  {stability;length;area;staircase;height;overpass;}

let tower_cash = Hashtbl.Poly.create();;

let update_serialized_tower_cash() =
  let open Yojson.Safe.Util in
  let open Yojson.Safe in
  let open Cache_pb in
  let m = (`String("sendSerializedCash")) in
  let filename =  send_to_tower_server m |> to_string in
  (* For some reason the filename still includes quotes, so we need to remove the first and last characters *)
  let filename = String.sub ~pos:1 ~len:(String.length filename - 2) filename in
  let c = 
    let ic = open_in filename in
    let len = in_channel_length ic in
    let bytes = Bytes.create len in 
    really_input ic bytes 0 len; 
    close_in ic; 
    Cache_pb.decode_tower_cash (Pbrt.Decoder.of_bytes bytes)
  in
  c.entries |> List.iter ~f:(fun e ->
      let plan = e.plan |> List.map ~f:(fun b -> (Int32.to_int_exn b.x10,
                                                  Int32.to_int_exn b.w10,
                                                  Int32.to_int_exn b.h10)) in
      (* Printf.eprintf "(ocaml received entry: %s)\n" *)
      (*   (plan |> List.map ~f:(fun (x ,w,h) -> Printf.sprintf "(%d,%d,%d)" x w h) |> join ~separator:";"); *)
      Hashtbl.Poly.set tower_cash ~key:plan ~data:{stability = e.stability;
                                                   length = e.length;
                                                   height = e.height;
                                                   area = e.area;
                                                   staircase = e.staircase;
                                                   overpass = e.overpass;})


let evaluate_tower ?n:(n=15) plan perturbation =
  (* center the tower *)
  let plan = center_tower plan in
  let key = plan in
  let open Yojson.Safe.Util in
  match Hashtbl.Poly.find tower_cash key with
  | Some(r) -> begin
      (* Printf.eprintf "(ocaml: hit %s)\n" *)
      (*   (key |> List.map ~f:(fun (a,b,c) -> Printf.sprintf "%d,%d,%d" a b c) |> join ~separator:"; "); *)
      (* flush_everything(); *)
      r
    end
  | None ->
    (* Printf.eprintf "(ocaml: miss %s)\n" *)
    (*   (key |> List.map ~f:(fun (a,b,c) -> Printf.sprintf "%d,%d,%d" a b c) |> join ~separator:"; "); *)
    (* flush_everything(); *)
      
    let plan = `List(plan |> List.map ~f:(fun (a,b,c) -> `List([`Int(a);
                                                                `Int(b);
                                                                `Int(c);]))) in
    let response = send_to_tower_server (`Assoc([("plan",plan);
                                                 ("n",`Int(n));
                                                 ("perturbation",`Float(perturbation))])) in
    let r = parse_tower_result response in
    Hashtbl.Poly.set tower_cash ~key:key ~data:r;
    r;;


register_special_task "tower" (fun extra ?timeout:(timeout = 0.001)
    (* ?stabilityThreshold:(stabilityThreshold=0.5) *)
    (* ~perturbation ~maximumStaircase ~maximumMass ~minimumLength ~minimumArea ~minimumHeight ~minimumOverpass *)
                                name task_type examples ->
                                Printf.eprintf "FIXME: tower task needs to be updated to use int\n";
                                assert false;
                                  
  assert (task_type = ttower @> ttower);
  assert (examples = []);

  let open Yojson.Basic.Util in
  
  let perturbation = extra |> member "perturbation" |> to_float in  
  let maximumStaircase = extra |> member "maximumStaircase" |> to_float in
  let maximumMass = extra |> member "maximumMass" |> to_float in 
  let minimumLength = extra |> member "minimumLength" |> to_float in 
  let minimumArea = extra |> member "minimumArea" |> to_float in 
  let minimumOverpass = extra |> member "minimumOverpass" |> to_float in 
  let minimumHeight = extra |> member "minimumHeight" |> to_float in 
  let stabilityThreshold = 0.5 in

  { name = name    ;
    task_type = task_type ;
    log_likelihood =
      (fun p ->
         let p = analyze_lazy_evaluation p in
         try
           match run_for_interval
                   timeout
                   (fun () -> run_lazy_analyzed_with_arguments p [fun s -> (s, [])] 0 |> snd)
           with
           | Some(p) ->
             let m = p |> List.map ~f:(fun (_,w,h) -> w*.h) |> List.fold_right ~f:(+.) ~init:0. in
             if m > maximumMass ||
                (* TODO *)
                (* tower_extent p > maximum_tower_extent || *)
                List.length p > maximum_number_of_blocks
             then log 0.0
             else 
               let result = evaluate_tower p perturbation in
               if result.height < minimumHeight
               || result.staircase > maximumStaircase
               || result.stability < stabilityThreshold
               || result.length < minimumLength
               || result.area < minimumArea
               || result.overpass < minimumOverpass
               then log 0.0
               else 50.0 *. log result.stability
           | _ -> log 0.0
         with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
              (* we have to be a bit careful with exceptions *)
              (* if the synthesized program generated an exception, then we just terminate w/ false *)
              (* but if the enumeration timeout was triggered during program evaluation, we need to pass the exception on *)
              | otherException -> begin
                  if otherException = EnumerationTimeout then raise EnumerationTimeout else log 0.
                end)

  })
;;

let simulate_without_physics plan =
  let overlaps (x,w,h) (x',y',w',h')  =
    let x1 = x - w/2 in
    let x2 = x + w/2 in
    let x1' = x' - w'/2 in
    let x2' = x' + w'/2 in
    if x1' > x2 || x1 > x2' then None else
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
                  (fun () -> run_lazy_analyzed_with_arguments p [fun s -> (s, [])] 0 |> snd)
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

let rec stress_test_client() =
  let open Yojson.Basic.Util in
  let m = (`String("doNothing")) in
  assert(send_to_tower_server m = `String("noop"));
  Printf.printf "+";
  flush_everything();
  stress_test_client()
;;

  
(* (lambda (tower_loopM 6 (lambda (lambda (3x1 $0))) $0)) *)
