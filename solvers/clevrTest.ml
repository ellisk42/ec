open Core

open Dreaming

open Program
open Enumeration
open Grammar
open Utils
open Timeout
open Type
    
open Yojson.Basic

let print_list obj_list = 
  obj_list |> List.map ~f: (fun obj -> 
    let (_, id) =  List.Assoc.find_exn obj "id" ~equal:(=) in
    let (_, color) =  List.Assoc.find_exn obj "color" ~equal:(=) in
    let (_, shape) =  List.Assoc.find_exn obj "shape" ~equal:(=) in
    let (_, material) =  List.Assoc.find_exn obj "material" ~equal:(=) in
    let (_, size) =  List.Assoc.find_exn obj "size" ~equal:(=) in
    (* let _ = Printf.eprintf "Id: %s \n" id in  *)
    Printf.eprintf "Color : %s | Shape: %s | Material: %s | Size: %s\n" color shape material size;
  );;

let test_program name raw input =
  let obj_list = List.hd_exn input in 
  let sorted_obj = sort_objs obj_list in
  let obj_1 = List.hd_exn sorted_obj in

  (* try 
    let (k, v) = List.Assoc.find_exn obj_1 "id" ~equal:(=) in
    Printf.eprintf "First %s id: out %d \n" name v
  with _ -> 
    Printf.eprintf "Error printing... \n" 
  in  *)
  let p = parse_program raw |> get_some in
  let p = analyze_lazy_evaluation p in
  let y = run_lazy_analyzed_with_arguments p input in
  (* Printf.eprintf "%s | out %s \n" name (Bool.to_string y);; *)

  print_list y;;
  (* Printf.eprintf "%s | out %s \n" name (Bool.to_string y);; *)

let run_job channel =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel channel in
  let request = j |> member "request" |> deserialize_type in
  let timeout = j |> member "timeout" |> to_float in
  let evaluationTimeout =
    try j |> member "evaluationTimeout" |> to_float
    with _ -> 0.001
  in
  let nc =
    try j |> member "CPUs" |> to_int
    with _ -> 1
  in
  let maximumSize =
    try j |> member "maximumSize" |> to_int
    with _ -> Int.max_value
  in
  let g = j |> member "DSL" in
  let g =
    try deserialize_grammar g |> make_dummy_contextual
    with _ -> deserialize_contextual_grammar g
  in
  let show_vars = 
    try j |> member "use_vars_in_tokenized" |> to_bool
    with _ -> false
  in
  let k =
    try Some(j |> member "special" |> to_string)
    with _ -> None
  in
  let k = match k with
    | None -> default_hash
    | Some(name) -> match Hashtbl.find special_helmholtz name with
      | Some(special) -> special
      | None -> (Printf.eprintf "Could not find special Helmholtz enumerator: %s\n" name; assert (false))
  in 
  let inputs = (j |> member "extras") in
  let unpacked_inputs : 'a list list = unpack_clevr inputs in
  let outputs = unpacked_inputs |> List.map ~f: (fun input -> 
    let raw =  "(lambda (clevr_eq_size (clevr_query_size (clevr_car $0)) clevr_small))" in
    let raw =  "(lambda (clevr_eq_objects (clevr_car $0) (clevr_car $0)))" in
    let raw = "(lambda (clevr_filter_size $0 clevr_large))" in
    let raw = "(lambda (clevr_filter_color $0 clevr_blue))" in
    let raw = "(lambda (clevr_same_size (clevr_car $0) $0))" in
    let raw = "(lambda (clevr_union (clevr_filter_size $0 clevr_large) (clevr_filter_shape $0 clevr_cube)))" in 
    let raw = "(lambda (clevr_intersect (clevr_filter_size $0 clevr_large) (clevr_filter_shape $0 clevr_cube)))" in 
    let raw = "(lambda (clevr_difference (clevr_filter_size $0 clevr_large) (clevr_filter_shape $0 clevr_cube)))" in 
    let raw = "(lambda (clevr_count (clevr_filter_size $0 clevr_large)))" in 
    let raw = "(lambda (clevr_eq_int (clevr_count (clevr_filter_size $0 clevr_large)) 3))"  in 
    let raw = "(lambda (clevr_gt? (clevr_count (clevr_filter_size $0 clevr_large)) 3))" in
    let raw = "(lambda (not (clevr_gt? (clevr_count (clevr_filter_size $0 clevr_large)) 3)))" in 
    let raw = "(lambda (clevr_map (clevr_transform_size clevr_large) $0))" in test_program "test" raw input
    ) in
  let message : json = 
    `List(
        [`Assoc([(* "behavior", behavior; *)
                "ll", `Float(1.0);
                "programs", `Float(1.0);
                "tokens", `Float(1.0);
                ])])
  in 
  message 
  

let _ = 
  run_job Pervasives.stdin 
  