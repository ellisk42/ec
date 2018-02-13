open Core.Std

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
        (* Printf.printf "About evaluate the log likelihood of program : %s =  %s\n" (string_of_type ty) (string_of_program p); *)
        (*        flush_everything(); *)
        let rec loop = function
          | [] -> true
          | (xs,y) :: e ->
            (* let xp : int list = magical y in *)
            (* xp |> List.iter ~f:(fun xpp -> Printf.printf "%d;" xpp);Printf.printf "\n"; *)
            (* flush_everything(); *)
            try
              match run_for_interval timeout (fun () -> Some(run_with_arguments p xs = y)) with
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
    ~timeout:timeout ?maximumFrontier:(maximumFrontier = 10) (t: task)
  =
  let open Timeout in
  
  let hits = ref [] in
  let budgetIncrement = 1. in
  let lower_bound = ref 0. in

  let total_count = ref 0 in

  let startTime = Time.now () in

  try
    while List.length (!hits) < maximumFrontier do
      let recent_count = ref 0 in
      enumerate_programs g empty_context (t.task_type) []
        (!lower_bound) (!lower_bound +. budgetIncrement)
        (fun p _ logPrior ->
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
                 Printf.printf "HIT %s w/ %s\n" (t.name) (string_of_program p)
               else ()
             end else  ());
      if verbose then
        Printf.printf "Enumerated %d programs satisfying %f < MDL <= %f\n"
          (!recent_count) (!lower_bound) (!lower_bound+.budgetIncrement)
      else ();
      lower_bound := budgetIncrement+. (!lower_bound);
      total_count := !total_count + !recent_count;
      if verbose then begin 
        Printf.printf "\tTotal time: %s. Total number of programs: %d.\n"
          (Time.diff (Time.now ()) startTime |> Core.Span.to_string)
          (!total_count);
        flush_everything();
      end else ()
    done;
    !hits
  with EnumerationTimeout -> !hits


  
