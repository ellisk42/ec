(*
test_clevr_primitives.ml | Author: Catherine Wong
Contains testing functionality for the CLEVR scene task primitives.
These tests are called from test_clevrPrimitivesOcaml.py
*)

open Core
open Utils
open Type
open Program
open Task
open Grammar

(** Loads tasks from StdIn. Expects a JSON object that contains:
  DSL : serialized grammar
  Tasks: List of 
    [ task JSON objects with:
      {
        "examples": list of [{"inputs": [list of inputs], "outputs": []}],
        "name" : task name,
        "request" : return type.
      }
    ]
 **)
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
        defaultTimeout
      end
  in
  
  (* Automatic differentiation parameters *)
  let maxParameters =
    try j |> member "maxParameters" |> to_int
    with _ -> 99
  in
  
  (** Unpacks CLEVR scenes recursively into a list of associations **)
  let rec unpack_clevr_input x =
    let open Yojson.Basic.Util in 
    try x |> to_assoc |> magical with _ ->
    try x |> to_list |> List.map ~f:unpack_clevr_input |> magical
    with _ -> raise (Failure "could not unpack clevr objects")
  in
  
  (** Unpacks CLEVR outputs by inferring the type. **)
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
  
  let tasks_and_programs_to_test = j |> member "tasks" |> to_list |> List.map ~f:(fun j -> 
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
           let special = j |> member "specialTask" |> to_string in
           match special |> Hashtbl.find task_handler with
           | Some(handler) -> handler (j |> member "extras")
           | None -> (Printf.eprintf " (ocaml) FATAL: Could not find handler for %s\n" special;
                      exit 1)
         with _ -> supervised_task) ~timeout:timeout name task_type examples
      in
      let raw_programs_to_test = j |> member "raw_programs_to_test" |> to_list |> List.map ~f: (fun json_program_string -> to_string json_program_string) 
      in 
      (task, raw_programs_to_test))
  in
  
  let verbose = try j |> member "verbose" |> to_bool      
    with _ -> false
  in
  
  (tasks_and_programs_to_test, g, verbose)

;;

(** Exports task results back to STD In. **)
let export_task_test_result tasks_and_programs did_tasks_succeed: string =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let serialization : Yojson.Basic.json =
    `Assoc(
          ("is_ocaml_test", `String("is_ocaml_test")) ::
          List.map2_exn tasks_and_programs did_tasks_succeed ~f: (fun(t,_) did_task_succeed -> 
             (t.name, `Bool(did_task_succeed))
           ))
  in pretty_to_string serialization
;;

(** Executes provided programs on a single task.
Takes: task object
      programs_to_test : [array of string programs]
 **)
let test_raw_programs_on_task task programs_to_test = 
  let did_all_succeed = List.map programs_to_test ~f: (fun program_to_test ->
    (Printf.eprintf "(ocaml) testing program: %s\n" program_to_test);
    let p = parse_program program_to_test |> get_some in
    let log_likelihood = task.log_likelihood p in
    let did_program_succeed = log_likelihood >= 0.0 in
    did_program_succeed
    )
  in
  let did_succeed = List.for_all ~f:(fun x -> x) did_all_succeed in
  did_succeed
;;

(** Executes all provided programs on the task. Returns [array of true] of length tasks_and_programs_to_test **)
let did_all_programs_succeed_on_tasks tasks_and_programs_to_test =
  let did_all_succeed = List.map tasks_and_programs_to_test ~f: (
    (fun(task, programs_to_test) -> test_raw_programs_on_task task programs_to_test)
    )
  in
  did_all_succeed
;;
  
(** Read in tasks from the main channel **)
let _ =
  let (tasks_and_programs_to_test, g, verbose) = load_problems Pervasives.stdin in
  let did_raw_programs_succeed = did_all_programs_succeed_on_tasks tasks_and_programs_to_test in
  export_task_test_result tasks_and_programs_to_test did_raw_programs_succeed |> print_string ;;