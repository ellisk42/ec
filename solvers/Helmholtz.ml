open Core

open Dreaming

open Program
open Grammar
open Type

open Yojson.Basic




let run_job channel =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel channel in
  let request = j |> member "request" |> deserialize_type in
  let timeout = j |> member "timeout" |> to_number in
  let evaluationTimeout =
    try j |> member "evaluationTimeout" |> to_number
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

let output_job ?maxExamples:(maxExamples=50000) result =
  (* let result = Hashtbl.to_alist result in *)
  let results =
    let l = List.length result in
    if l < maxExamples then result else
      let p = (maxExamples |> Float.of_int)/.(l |> Float.of_int) in
      result |> List.filter ~f:(fun _ -> Float.(<) (Random.float 1.) p)
  in
  let message : Yojson.Basic.t =
    `List(results |> List.map ~f:(fun (_behavior, (l,ps)) ->
        `Assoc([(* "behavior", behavior; *)
                "ll", `Float(l);
                "programs", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_program)))])))
  in
  message

let _ : unit =
  run_job Stdlib.stdin |> remove_bad_dreams |> output_job |> to_channel Stdlib.stdout
