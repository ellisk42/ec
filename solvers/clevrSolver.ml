open Core

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

  let timeout = try
      j |> member "programTimeout" |> to_float
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
  
  let rec unpack_clevr_input x =
    let open Yojson.Basic.Util in 
    try x |> to_assoc |> magical with _ ->
    try x |> to_list |> List.map ~f:unpack_clevr_input |> magical
    with _ -> raise (Failure "could not unpack clevr objects")
  in
  
  (** Assumes list must be a CLEVR object **)
  let unpack_clevr_output y t = 
    let open Yojson.Basic.Util in
    try
      match t with
      | TCon("int",[],_) ->  magical (y |> to_int)
      | TCon("bool",[],_) ->  magical (y |> to_bool)
      | TCon("list",[t'],_) -> 
         y |> to_list |> List.map ~f: unpack_clevr_input |> magical 
      | _ ->  magical (y |> to_string)
    with _-> raise (Failure "could not unpack clevr output")
  in
  let tf = j |> member "tasks" |> to_list |> List.map ~f:(fun j -> 
      let e = j |> member "examples" |> to_list in
      let task_type = j |> member "request" |> deserialize_type in 
      let return_type = return_of_type task_type in 
      let examples = e |> List.map ~f:(fun ex -> 
        let unpacked_inputs = ex |> member "inputs" |> to_list |> List.map ~f:unpack_clevr_input in
        let output = ex |> member "output" in 
        let unpacked_outputs = unpack_clevr_output output return_type in 
        (unpacked_inputs, unpacked_outputs)) in
      let maximum_frontier = j |> member "maximumFrontier" |> to_int in
      let name = j |> member "name" |> to_string in

      let task =
        (try
          (** We use a special handler for evaluating return types with a CLEVR object list **)
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
    try j |> member "lowerBound" |> to_float
    with _ -> 0.
  in

  let upperBound =
    try j |> member "upperBound" |> to_float
    with _ -> 99.
  in

  let budgetIncrement =
    try j |> member "budgetIncrement" |> to_float
    with _ -> 1.
  in

  let timeout = j |> member "timeout" |> to_float in
  let nc =
    try
      j |> member "nc" |> to_int 
    with _ -> 1
  in
  (tf,g,
   lowerBound,upperBound,budgetIncrement,
   maxParameters,
   nc,timeout,verbose)

let export_frontiers number_enumerated tf solutions: string =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let serialization : Yojson.Basic.json =
    `Assoc(("number_enumerated",`Int(number_enumerated)) ::
           List.map2_exn tf solutions ~f:(fun (t,_) ss ->
        (t.name, `List(ss |> List.map ~f:(fun s ->
             `Assoc([("program", `String(s.hit_program));
                     ("time", `Float(s.hit_time));
                     ("logLikelihood", `Float(s.hit_likelihood));
                     ("logPrior", `Float(s.hit_prior));
                     ("tokens", `String(s.hit_tokens))]))))))
  in pretty_to_string serialization
;;


let _ =
  let (tf,g,
       lowerBound,upperBound,budgetIncrement,
       mfp,
     nc,timeout, verbose) =
    load_problems Pervasives.stdin in
  let solutions, number_enumerated =
    enumerate_for_tasks ~maxFreeParameters:mfp ~lowerBound:lowerBound ~upperBound:upperBound ~budgetIncrement:budgetIncrement
    ~verbose:verbose ~nc:nc ~timeout:timeout g tf
  in
  export_frontiers number_enumerated tf solutions |> print_string ;;