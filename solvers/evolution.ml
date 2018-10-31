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

open PolyValue

exception AncestorFailure;;

let evolution_enumeration (behavior_hash : program -> PolyList.t option) ?nc:(nc=1) g request ~ancestor
    ~timeout ~maximumSize =

  let rec substitute_ancestor = function
    | Primitive(_,"ancestor",_) -> if is_some ancestor then get_some ancestor else raise AncestorFailure
    | Abstraction(b) -> Abstraction(substitute_ancestor b)
    | Apply(f,x) -> Apply(substitute_ancestor f, substitute_ancestor x)
    | anything_else -> anything_else
  in
  
  let behavior_hash = behavior_hash % substitute_ancestor
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
  let ancestor =
    try j |> member "ancestor" |> to_string |> parse_program
    with _ -> None
  in
  ignore(primitive "ancestor" request ());
  let g = j |> member "DSL" |> deserialize_contextual_grammar in  

  (* this will remove the ancestor primitive if we do not have an ancestor *)
  let strip_ancestor g =
    {g with library =
              g.library |> List.filter ~f:(fun (p,_,_,_) -> match (ancestor, p) with
                  (* do not have an ancestor - strip out the ancestor primitive *)
                  | None, Primitive(_,"ancestor",_) -> false
                  | _ -> true)}
  in
  let g = {no_context = g.no_context |> strip_ancestor;
           variable_context = g.no_context |> strip_ancestor;
           contextual_library = g.contextual_library |>
                                List.map ~f:(fun (p,gs) ->
                                    (p,gs |> List.map ~f:strip_ancestor))}
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
    `List(Hashtbl.to_alist result |> List.map ~f:(fun (behavior, (l,ps)) ->
        `Assoc([(* "behavior", behavior; *)
                "ll", `Float(l);
                "programs", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_program)))])))
  in 
  message

let () =
  run_job Pervasives.stdin |> output_job |> to_channel Pervasives.stdout
