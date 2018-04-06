open Core

open Timeout
open Utils
open Type
open Program
open Enumeration
open Grammar
open Differentiation

type task =
  { name: string; task_type: tp;
    log_likelihood: program -> float;
  }

exception EnumerationTimeout

let supervised_task ?timeout:(timeout = 0.001) name ty examples =
  { name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun p ->
        let p = analyze_lazy_evaluation p in
        let rec loop = function
          | [] -> true
          | (xs,y) :: e ->
            try
              match run_for_interval
                      timeout
                      (fun () -> run_lazy_analyzed_with_arguments p xs = y)
              with
                | Some(true) -> loop e
                | _ -> false
            with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                 (* we have to be a bit careful with exceptions *)
                 (* if the synthesized program generated an exception, then we just terminate w/ false *)
                 (* but if the enumeration timeout was triggered during program evaluation, we need to pass the exception on *)
                 | otherException -> begin
                     if otherException = EnumerationTimeout then raise EnumerationTimeout else false
                   end
        in
        if loop examples
          then 0.0
          else log 0.0)
  }

let differentiable_task
    ?parameterPenalty:(parameterPenalty=0.)
    ?lossThreshold:(lossThreshold=None)
    ?maxParameters:(maxParameters=100)
    ?timeout:(timeout = 0.001) name ty examples =
  (* Process the examples and wrap them inside of placeholders *)
  let (argument_types, return_type) = arguments_and_return_of_type ty in
  let examples = examples |> List.map ~f:(fun (xs,y) ->
      (List.map2_exn argument_types xs ~f:placeholder_data,
      placeholder_data return_type y))
  in
  let loss = polymorphic_sse return_type in
  { name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun expression ->
        let (p,parameters) = replace_placeholders expression in
        if List.length parameters > maxParameters then log 0. else 
        (* Printf.eprintf "%s has d=%d\n" (string_of_program expression) (List.length parameters); *)
        let p = analyze_lazy_evaluation p in
        (* Returns loss *)
        let rec loop : 'a list -> Differentiation.variable option = function
          | [] -> Some(~$ 0.)
          | (xs,y) :: e ->
            try
              match run_for_interval
                      timeout
                      (fun () -> run_lazy_analyzed_with_arguments p xs) with
              | None -> None
              | Some (prediction) ->
                match loop e with
                  | None -> None
                  | Some(later_loss) ->
                    try Some(loss y prediction +& later_loss)
                    with DifferentiableBadShape -> None
            with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                 | otherException -> begin
                     if otherException = EnumerationTimeout then raise EnumerationTimeout else None
                   end
        in
        match loop examples with
        | None -> log 0.0
        | Some(l) ->
          let n = List.length examples |> Int.to_float in
          let d = List.length parameters |> Int.to_float in
          let l = l *& (~$ (1. /. n)) in
          let l = run_optimizer (rprop ~lr:0.1 ~decay:0.5 ~grow:1.2)
              ~update:0
              ~iterations:(if List.length parameters = 0 then 0 else 100)
              parameters l
          in
          (* Printf.eprintf "%s has l=%f\n" (string_of_program expression) l;
           * flush_everything(); *)
          match lossThreshold with
          | None -> 0. -. d*.parameterPenalty -. n *. l
          | Some(t) ->
            if l < t then 0. -. d*.parameterPenalty else log 0.)
  }


let keep_best_programs_in_frontier (k : int) (f : frontier) : frontier =
  {request = f.request;
   programs =  List.sort ~cmp:(fun (_,a) (_,b) -> if a > b then -1 else 1) f.programs |> flip List.take k }

(* Takes a frontier and a task. Ads in the likelihood on the task to
   the frontier and removes things that didn't hit the task *)
let score_programs_for_task (f:frontier) (t:task) : frontier =
  {request = f.request;
   programs = f.programs |> List.filter_map ~f:(fun (program, descriptionLength) ->
       let likelihood = t.log_likelihood program in
       if likelihood > -0.1 then 
         Some((program, descriptionLength +. likelihood))
       else None)
  }

let enumerate_for_task (g: grammar) ?verbose:(verbose = true)
    ?budgetIncrement:(budgetIncrement = 1.)
    ?lowerBound:(lowerBound = 0.)
    ?upperBound:(upperBound = 99.)
    ?maximumFrontier:(maximumFrontier = 10) (t: task)
  =

  (* Store the hits in a priority queue *)
  (* We will only ever maintain maximumFrontier best solutions *)
  let hits =
    Heap.create
      ~cmp:(fun (_,lp1,ll1,_) (_,lp2,ll2,_) ->
              let c = (lp1+.ll1) -. (lp2+.ll2) in
              if c < 0. then -1 else if c > 0. then 1 else 0) () in
  let lower_bound = ref lowerBound in

  let total_count = ref 0 in

  let programs_explored = ref 0 in

  let startTime = Time.now () in

  let listenForSIGUSR1 =
    Caml.Sys.Signal_handle (
      fun _ -> raise EnumerationTimeout
    ) in

  Caml.Sys.set_signal Caml.Sys.sigusr1 listenForSIGUSR1 ;

  try
    while Heap.length hits < maximumFrontier
       && !lower_bound +. budgetIncrement <= upperBound
    do
      let recent_count = ref 0 in
      enumerate_programs g (t.task_type)
        (!lower_bound) (!lower_bound +. budgetIncrement)
        (fun p logPrior ->
           incr programs_explored ;
           let mdl = 0.-.logPrior in

           assert( !lower_bound <= mdl);
           assert( mdl < budgetIncrement+.(!lower_bound));

           incr recent_count ;

           let logLikelihood = t.log_likelihood p in
           if is_valid logLikelihood then begin
             let dt = Time.abs_diff startTime (Time.now ())
                      |> Time.Span.to_sec in
             Heap.add hits (p,logPrior,logLikelihood,dt) ;
             if Heap.length hits > maximumFrontier then Heap.remove_top hits ;
             if verbose then
               Printf.eprintf
                "\t(ocaml) HIT %s w/ %s\n" (t.name) (string_of_program p)
             end) ;
      if verbose then
        Printf.eprintf "\t(ocaml) For %s : %s, enumerated %d programs satisfying %f < MDL <= %f\n"
          (t.name) (t.task_type |> string_of_type) (!recent_count) (!lower_bound) (!lower_bound+.budgetIncrement)
      else ();
      lower_bound := budgetIncrement+. (!lower_bound);
      total_count := !total_count + !recent_count;
      if verbose then begin
        Printf.eprintf "\t(ocaml) For %s: Total time: %s. Total number of programs: %d.\n"
          (t.name)
          (Time.diff (Time.now ()) startTime |> Time.Span.to_string)
          (!total_count);
        flush_everything();
      end else ()
    done ;
    (Heap.to_list hits, !programs_explored)
  with EnumerationTimeout -> (Heap.to_list hits, !programs_explored)
