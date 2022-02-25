open Core

open Probabilistic_grammar
open Physics
open Pregex
open Tower
(* open Vs *)
open Differentiation
open TikZ
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Task
open FastType

let load_problems channel =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel channel in
  let g = j |> member "DSL" in
  let g =
    try deserialize_grammar g |> make_dummy_contextual
    with _ -> deserialize_contextual_grammar g
  in

  let unrolled_grammar =
    try Some(j |> member "PCFG" |> deserialize_PCFG)
    with _ -> None
  in 

  let timeout = try
      j |> member "programTimeout" |> to_number
    with _ ->
      begin
        let defaultTimeout = 0.1 in
        Printf.eprintf
          "\t(ocaml) WARNING: programTimeout not set. Defaulting to %f.\n"
          defaultTimeout ;
        defaultTimeout
      end
  in

  (* Automatic differentiation parameters *)
  let maxParameters =
    try j |> member "maxParameters" |> to_int
    with _ -> 99
  in


  let rec unpack x =
    try magical (x |> to_int) with _ ->
    try magical (x |> to_number) with _ ->
    try magical (x |> to_bool) with _ ->
    try
      let v = x |> to_string in
      if String.length v = 1 then magical v.[0] else magical v
    with _ ->
    try
      x |> to_list |> List.map ~f:unpack |> magical
    with _ -> raise (Failure "could not unpack")
  in

  let tf = j |> member "tasks" |> to_list |> List.map ~f:(fun j -> 
      let e = j |> member "examples" |> to_list in
      let task_type = j |> member "request" |> deserialize_type in 
      let examples = e |> List.map ~f:(fun ex -> (ex |> member "inputs" |> to_list |> List.map ~f:unpack,
                                                  ex |> member "output" |> unpack)) in
      let maximum_frontier = j |> member "maximumFrontier" |> to_int in
      let name = j |> member "name" |> to_string in

      let task =
        (try
           let special = j |> member "specialTask" |> to_string in
           match special |> Hashtbl.find task_handler with
           | Some(handler) -> handler (j |> member "extras")
           | None -> (Printf.eprintf " (ocaml) FATAL: Could not find handler for %s\n" special;
                      exit 1)
         with _ -> supervised_task) ~timeout:timeout name task_type examples
      in 
      (task, maximum_frontier))
  in

  (* Ensure that all of the tasks have the same type *)
  (* let most_specific_type = unify_many_types (tf |> List.map ~f:(fun (t,_) -> t.task_type)) in
   * let tf = tf |> List.map ~f:(fun (t,f) -> ({t with task_type=most_specific_type},f)) in *)

  let verbose = try j |> member "verbose" |> to_bool      
    with _ -> false
  in
  
  let _ = try
      shatter_factor := (j |> member "shatter" |> to_int)
    with _ -> ()
  in


  let lowerBound =
    try j |> member "lowerBound" |> to_number
    with _ -> 0.
  in

  let upperBound =
    try j |> member "upperBound" |> to_number
    with _ -> 99.
  in

  let budgetIncrement =
    try j |> member "budgetIncrement" |> to_number
    with _ -> 1.
  in

  let timeout = j |> member "timeout" |> to_number in
  let nc =
    try
      j |> member "nc" |> to_int 
    with _ -> 1
  in
  (tf,g,unrolled_grammar, 
   lowerBound,upperBound,budgetIncrement,
   maxParameters,
   nc,timeout,verbose)

let export_frontiers number_enumerated tf solutions : string =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let serialization : Yojson.Basic.json =
    `Assoc(("number_enumerated",`Int(number_enumerated)) ::
           List.map2_exn tf solutions ~f:(fun (t,_) ss ->
        (t.name, `List(ss |> List.map ~f:(fun s ->
             `Assoc([("program", `String(s.hit_program));
                     ("time", `Float(s.hit_time));
                     ("logLikelihood", `Float(s.hit_likelihood));
                     ("logPrior", `Float(s.hit_prior))]))))))
  in pretty_to_string serialization
;;


let _ =

  let (tf,g,unrolled,
       lowerBound,upperBound,budgetIncrement,
       mfp,
     nc,timeout, verbose) =
    load_problems Pervasives.stdin in
  let quick_tasks = tf |> List.map ~f:(fun (t, k) ->
      ({name= t.name; task_type= t.task_type;
           log_likelihood= (fun _ -> log 0.)},k )) in
  let slow_tasks = tf |> List.map ~f:(fun (t, k) ->
      ({name= t.name; task_type= t.task_type;
           log_likelihood= (fun p -> t.log_likelihood p; log 0.)} ), k) in
  flush_everything();
  let _T,_ub = 5.,30. in

  let backend=
    match unrolled with
    | None -> 
      let traditional_backend lowerBound upperBound ~final =
        enumerate_programs ~final g (List.hd_exn tf |> fst).task_type lowerBound upperBound ~maxFreeParameters:mfp ~nc
      in traditional_backend
    | Some(unrolled) -> 
      let new_backend lower_bound upper_bound ~final continuation =
        (* bounded_recursive_enumeration *) (* bottom_up_enumeration *)
        dynamic_programming_enumeration
        ~factor:2
          ~lower_bound ~upper_bound g.variable_context (List.hd_exn tf |> fst).task_type (* unrolled *)
          (fun p l -> continuation p l);
        [final()]
      in new_backend
  in 
  
  (* let progress_without_evaluation = *)
  (* enumerate_for_tasks traditional_backend ~lowerBound:0. ~upperBound:_ub ~budgetIncrement:budgetIncrement *)
  (*     ~verbose:true ~nc ~timeout:_T quick_tasks |> snd in *)

  (* flush_everything(); *)
  (* let progress_with_evaluation = enumerate_for_tasks traditional_backend ~lowerBound:0. ~upperBound:_ub ~budgetIncrement:budgetIncrement *)
  (*     ~verbose:true ~nc ~timeout:_T slow_tasks |> snd in *)
  (* flush_everything(); *)
  (* let enumerations_per_second=((Float.of_int progress_without_evaluation)/._T) in *)
  (* let evaluations_per_second= 1./.(_T/.(Float.of_int progress_without_evaluation) -. _T/.(Float.of_int progress_with_evaluation)) in *)

  (* if true || progress_without_evaluation > progress_with_evaluation then ( *)
  (*   (\* unrolled |> show_probabilistic |> Printf.eprintf "%s\n"; *\) *)
  (*   let starting = Unix.time() in *)
  (*   set_enumeration_timeout _T; *)
  (*   let fast_enumerated= *)
  (*     let count=ref 0 in *)
  (*     bottom_up_enumeration *)
  (*       (\* bounded_recursive_enumeration *\) *)
  (*       ~lower_bound:0. ~upper_bound:_ub unrolled *)
  (*   (fun p l -> incr count); *)
  (*   !count *)
  (*     (\* enumerate_for_tasks new_backend ~lowerBound:0. ~upperBound:_ub ~budgetIncrement:budgetIncrement *\) *)
  (*     (\*   ~verbose:true ~timeout:_T quick_tasks ~nc |> snd *\) *)
  (*   in *)
  (*   let fast_enumerated_time = Unix.time()-.starting in  *)
    
  (*   Printf.eprintf "How many can we enumerate if we DO NOT evaluate programs? %d\n" progress_without_evaluation; *)
  (*   Printf.eprintf "How many can we enumerate if we DO evaluate programs? %d\n" progress_with_evaluation; *)
  (*     Printf.eprintf "How many can we enumerate with FAST %d/%fs\n" fast_enumerated fast_enumerated_time; *)
  (*   Printf.eprintf "Enumerated programs per second: %f\n" enumerations_per_second; *)
  (*   Printf.eprintf "PCFG Enumerated programs per second: %f\n" *)
  (*     ((Float.of_int fast_enumerated) /. _T); *)
  (*   Printf.eprintf "Evaluations per second: %f\n" evaluations_per_second); *)

  let solutions, number_enumerated =
    enumerate_for_tasks backend ~lowerBound:lowerBound ~upperBound:upperBound ~budgetIncrement:budgetIncrement
    ~verbose:verbose ~timeout:timeout tf ~nc
  in
  export_frontiers number_enumerated tf solutions |> print_string ;;

(* let tune_differentiation () = *)
(*   let (tf,g, *)
(*        lowerBound,upperBound,budgetIncrement, *)
(*        mfp, *)
(*      nc,timeout, verbose) = *)
(*     load_problems  Pervasives.stdin in *)
(*   if List.exists tf ~f:(fun (t,_) -> t.task_type = ttower) then update_tower_cash() else (); *)
(*   (\* "(-6.8x + 4.7)/[(x + 4.5)]" *\) *)
(*   let p = parse_program "(lambda (/. (+. REAL (\*. REAL $0)) (+. $0 REAL)))" |> get_some in *)
(*   match tf with *)
(*     | [(t,_)] -> Printf.eprintf "%f\n" (t.log_likelihood p) *)
(*     | _ -> failwith "ERROR: no task were given at all" *)
(* ;; *)
