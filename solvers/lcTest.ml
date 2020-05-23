open Core
open Gc
open Physics
open Pregex
open Tower
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar

(* open Eg *)
open Versions


(** JSON loop -- can't import from compression since we will overwrite () **)
let () =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let j =
    if Array.length Sys.argv > 1 then
      (assert (Array.length Sys.argv = 2);
       Yojson.Basic.from_file Sys.argv.(1))
    else 
      Yojson.Basic.from_channel Pervasives.stdin
  in
  let g = j |> member "DSL" |> deserialize_grammar |> strip_grammar in
  let topK = j |> member "topK" |> to_int in
  let topI = j |> member "topI" |> to_int in
  let bs = j |> member "bs" |> to_int in
  let arity = j |> member "arity" |> to_int in
  let aic = j |> member "aic" |> to_float in
  let pseudoCounts = j |> member "pseudoCounts" |> to_float in
  let structurePenalty = j |> member "structurePenalty" |> to_float in

  verbose_compression := (try
      j |> member "verbose" |> to_bool
                          with _ -> false);

  factored_substitution := (try
                              j |> member "factored_apply" |> to_bool
                            with _ -> false);
  if !factored_substitution then Printf.eprintf "Using experimental new factored representation of application version space.\n";

  collect_data := (try
                     j |> member "collect_data" |> to_bool
                   with _ -> false) ;
  if !collect_data then verbose_compression := true;

  
  let inline = (try
                  j |> member "inline" |> to_bool
                with _ -> true)
  in 

  let nc =
    try j |> member "CPUs" |> to_int
    with _ -> 1
  in
  
  (** Get the language alignments **)
  let language_alignments = (try
                  j |> member "language_alignments" |> to_list 
                with _ -> [])
  in
  let _ = Printf.eprintf "Found %d alignments; \n" (List.length language_alignments) in 
  let language_alignments = language_alignments |> List.map ~f:deserialize_alignment in

  let iterations = try
      j |> member "iterations" |> to_int
    with _ -> 1000
  in
  Printf.eprintf "Compression backend will run for most %d iterations\n"
    iterations;
  flush_everything();

  let frontiers = j |> member "frontiers" |> to_list |> List.map ~f:deserialize_frontier in
  
  (* let _ = language_alignments |> List.map ~f: align_to_string in
  
  let (p, _, _, _) = List.hd_exn g.library in 
  let _ = Printf.eprintf "Primitive: %s\n" (string_of_program p) in
  (* let _ = all_rewrite_score p language_alignments  *)
  
  let initial_score = all_initial_score language_alignments in 
  let _ = g.library |> List.map ~f: (
    fun (p, _, _, _) -> 
       let new_score = all_rewrite_score p language_alignments in
       let _ = Printf.eprintf "Primitive: %s\n" (string_of_program p) in
       let _ = if new_score > initial_score then Printf.eprintf "Improved score from %f to %f \n" initial_score new_score  
       in p
    )
  in
  let frontiers = j |> member "frontiers" |> to_list |> List.map ~f:deserialize_frontier in

  let iterations = try
      j |> member "iterations" |> to_int
    with _ -> 1000
  in
  Printf.eprintf "Compression backend will run for most %d iterations\n"
    iterations;
  flush_everything(); *)

  (* let g, frontiers =
    if aic > 500. then
      (Printf.eprintf "AIC is very large (over 500), assuming you don't actually want to do DSL learning!";
       g, frontiers)
    else compression_loop ~inline ~iterations ~nc ~topK ~aic ~structurePenalty ~pseudoCounts ~arity ~topI ~bs g frontiers in  *)
      

  (* let g, frontiers =
    if aic > 500. then
      (Printf.eprintf "AIC is very large (over 500), assuming you don't actually want to do DSL learning!";
       g, frontiers)
    else compression_loop ~inline ~iterations ~nc ~topK ~aic ~structurePenalty ~pseudoCounts ~arity ~topI ~bs g frontiers language_alignments in  *)
      

  let j = `Assoc(["DSL",serialize_grammar g;
                  "frontiers",`List(frontiers |> List.map ~f:serialize_frontier)])
  in
  pretty_to_string j |> print_string