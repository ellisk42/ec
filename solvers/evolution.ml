open Core

open Pregex
open Program
open Enumeration
open Grammar
open Utils
open Timeout
open Type
open Tower
open Dreaming

open Yojson.Basic


let evolution_enumeration (behavior_hash : program -> (int*(json list)) option) ?nc:(nc=1) g request ~ancestor
    ~timeout ~maximumSize =
  let request = match ancestor with
      None -> request
    | Some(_) -> request @> request
  in

  let behavior_hash = match ancestor with
    | None -> behavior_hash
    | Some(a) -> fun p -> behavior_hash (Apply(p,a))
  in

  helmholtz_enumeration behavior_hash ~nc g request ~timeout ~maximumSize



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
  let g = j |> member "DSL" |> deserialize_contextual_grammar in

  let ancestor =
    try j |> member "ancestor" |> to_string |> parse_program
    with _ -> None
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

  evolution_enumeration ~nc:nc (k ~timeout:evaluationTimeout request (j |> member "extras")) g request ~ancestor ~timeout ~maximumSize

let output_job result =
  let open Yojson.Basic.Util in
  let message : json = 
    `List(Hashtbl.to_alist result |> List.map ~f:(fun ((_, behavior), (l,ps)) ->
        `Assoc([(* "behavior", behavior; *)
                "ll", `Float(l);
                "programs", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_program)))])))
  in 
  message

let () =
  run_job Pervasives.stdin |> output_job |> to_channel Pervasives.stdout
