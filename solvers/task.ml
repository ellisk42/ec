open Core

open Timeout
open Utils
open Type
open Program
open Enumeration
open Grammar


type task =
  { name: string; task_type: tp;
    log_likelihood: program -> float;
    task_features: float list;
  }

let supervised_task ?timeout:(timeout = 0.001) ?features:(features = []) name ty examples =
  { name = name;
    task_features = features;
    task_type = ty;
    log_likelihood = (fun p ->
        let p = analyze_lazy_evaluation p in
        let rec loop = function
          | [] -> true
          | (xs,y) :: e ->
            try
              match run_for_interval timeout (fun () -> run_lazy_analyzed_with_arguments p xs = y) with
              | Some(true) -> loop e
              | _ -> false
            with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                 | _ -> false
        in
        if loop examples
        then 0.0
        else log 0.0)
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

exception EnumerationTimeout;;
let enumerate_for_task (g: grammar) ?verbose:(verbose = true)
    ?budgetIncrement:(budgetIncrement = 1.)
    ?lowerBound:(lowerBound = 0.)
    ?upperBound:(upperBound = 99.)
    ~timeout:timeout ?maximumFrontier:(maximumFrontier = 10) (t: task)
  =
  
  let hits = ref [] in
  let lower_bound = ref lowerBound in

  let total_count = ref 0 in

  let programs_explored = ref 0 in

  let startTime = Time.now () in

  try
    while List.length (!hits) < maximumFrontier && !lower_bound +. budgetIncrement <= upperBound do
      let recent_count = ref 0 in
      enumerate_programs g (t.task_type)
        (!lower_bound) (!lower_bound +. budgetIncrement)
        (fun p logPrior ->
           incr programs_explored;
           let df = Time.diff (Time.now ()) startTime |> Time.Span.to_sec in
           if df > (Float.of_int timeout) then raise EnumerationTimeout else 
             let mdl = 0.-.logPrior in
             assert( !lower_bound < mdl);
             assert( mdl <= budgetIncrement+.(!lower_bound));

             incr recent_count;
             let logLikelihood = t.log_likelihood p in
             if is_valid logLikelihood then begin 
               hits := (p,logPrior,logLikelihood,df) :: !hits;
               if verbose then  
                 Printf.eprintf "\t(ocaml) HIT %s w/ %s\n" (t.name) (string_of_program p)
               else ()
             end else  ());
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
    done;
    (!hits, !programs_explored)
  with EnumerationTimeout -> (!hits, !programs_explored)


  
