open Core.Std

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Compression
open Sfg


let rec exploration_compression
    (tasks : task list)
    (g : grammar)
    (frontier_size : int)
    ?keepTheBest:(keepTheBest = 1)
    ?arity:(arity = 0)
    (* lambda: penalty on the size of the structure of the grammar *)
    ?lambda:(lambda = 2.)
    (* alpha: pseudo counts for parameter estimation of grammar *)
    ?alpha:(alpha = 1.)
    (* beta: coefficient of AIC penalty *)
    ?beta:(beta = 1.)
    (iterations : int)
  : grammar =
  if iterations = 0 then g else
    let frontiers = enumerate_solutions_for_tasks g tasks frontier_size ~keepTheBest:keepTheBest in

    Printf.printf "Hit %d/%d of the tasks with a frontier of size %d.\n"
      (frontiers |> List.filter ~f:(fun f -> List.length f.programs > 0) |> List.length)
      (List.length frontiers)
      frontier_size;

    Out_channel.flush stdout;

    (*     estimate_categorized_fragment_grammar (fragment_grammar_of_grammar g) frontiers; *)

    let fragments = time_it "Proposing fragments" @@ fun _ -> propose_fragments_from_frontiers arity frontiers in

    Printf.printf "Proposed %d fragments.\n" (List.length fragments);

    (*     fragments |> List.iter ~f:(fun f -> Printf.printf "FRAGMENT\t%s\n" (string_of_fragment f)); *)

    Out_channel.flush stdout;
    
    let gf = time_it "Induced grammar" @@ fun _ ->
      induce_fragment_grammar ~lambda:lambda ~alpha:alpha ~beta:beta
        fragments frontiers (fragment_grammar_of_grammar g) in

    Out_channel.flush stdout;

    let gp = grammar_of_fragment_grammar gf in

    Printf.printf "GRAMMAR\n%s\n" (string_of_grammar gp);

    Out_channel.flush stdout;

    exploration_compression tasks gp frontier_size ~keepTheBest:keepTheBest (iterations - 1)
    
