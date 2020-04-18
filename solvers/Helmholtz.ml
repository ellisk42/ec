open Core

open Dreaming

open Pregex
open Program
open Enumeration
open Grammar
open Utils
open Timeout
open Type
open Tower
    
open Yojson.Basic

    
    

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

  helmholtz_enumeration ~nc:nc (k ~timeout:evaluationTimeout request (j |> member "extras")) g request ~timeout ~maximumSize

let output_job ?maxExamples:(maxExamples=50000) ?show_vars:(show_vars=false) result =
  let open Yojson.Basic.Util in
  (* let result = Hashtbl.to_alist result in *)
  let results =
    let l = List.length result in
    if l < maxExamples then result else
      let p = (maxExamples |> Float.of_int)/.(l |> Float.of_int) in
      result |> List.filter ~f:(fun _ -> Random.float 1. < p)
  in
  let message : json = 
    `List(results |> List.map ~f:(fun (behavior, (l,ps)) ->
        `Assoc([(* "behavior", behavior; *)
                "ll", `Float(l);
                "programs", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_program)));  
                "tokens", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_tokens show_vars)));
                ])))
  in 
  message

let _ = 
  run_job Pervasives.stdin |> remove_bad_dreams |> output_job |> to_channel Pervasives.stdout
