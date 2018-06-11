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
            with (* We have to be a bit careful with exceptions if the
                  * synthesized program generated an exception, then we just
                  * terminate w/ false but if the enumeration timeout was
                  * triggered during program evaluation, we need to pass the
                  * exception on
                  *)
              | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
              | EnumerationTimeout  -> raise EnumerationTimeout
              | _                   -> false
        in
        if loop examples
          then 0.0
          else log 0.0)
  }

let turtle_task ?timeout:(timeout = 0.001) name ty examples =
  match examples with
  | [(_,y)] ->
    let by = Bigarray.(Array1.of_array int8_unsigned c_layout (Array.of_list y))
    in
    { name = name    ;
      task_type = ty ;
      log_likelihood =
        (fun p ->
          let p = analyze_lazy_evaluation p in
          print_endline "so far so good 1" ;
          if (try begin
                match run_for_interval
                        timeout
                        (fun () ->
                          print_endline "so far so good 2" ;
                          let x = run_lazy_analyzed_with_arguments p [] in
                          print_endline "so far so good 3" ;
                          LogoLib.LogoInterpreter.pp_turtle x ;
                          print_endline "so far so good 4" ;
                          let bx = LogoLib.LogoInterpreter.turtle_to_array x 28 in
                          print_endline "so far so good 5" ;
                          bx = by)
                with 
                  | Some(true) -> true
                  | _ -> false
              end
              with (* We have to be a bit careful with exceptions if the
                    * synthesized program generated an exception, then we just
                    * terminate w/ false but if the enumeration timeout was
                    * triggered during program evaluation, we need to pass the
                    * exception on
                    *)
                | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                | EnumerationTimeout  -> raise EnumerationTimeout
                | _                   -> false)
            then 0.0
            else log 0.0)
    }
  | _ -> failwith "not a turtle task"

let differentiable_task
  ?temperature:(temperature=1.)
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
                    try Some(loss prediction y +& later_loss)
                    with DifferentiableBadShape -> None
            with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                 | EnumerationTimeout  -> raise EnumerationTimeout
                 | _                   -> None
        in
        match loop examples with
        | None -> log 0.0
        | Some(l) ->
          let n = List.length examples |> Int.to_float in
          let d = List.length parameters |> Int.to_float in
          let l = l *& (~$ (1. /. n)) in
          let l = restarting_optimize (rprop ~lr:0.5 ~decay:0.5 ~grow:1.2)
              ~attempts:300
              ~update:0
              ~iterations:(if List.length parameters = 0 then 0 else 50)
              parameters l
          in
          (* Printf.eprintf "%s has l=%f\n" (string_of_program expression) l;
           * flush_everything(); *)
          match lossThreshold with
          | None -> 0. -. d*.parameterPenalty -. n *. l /. temperature
          | Some(t) ->
            if l < t then 0. -. d*.parameterPenalty else log 0.)
  }

let constant_task
    ?parameterPenalty:(parameterPenalty=0.)
    ?maxParameters:(maxParameters=100)
    ?timeout:(timeout = 0.001)
    ~stringConstants
    name ty examples =
  let stringConstants : char list list = stringConstants in
  let lc = log (26.*.2.+.10.) in
  let lc = 0.-.lc in
  
  { name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun expression ->
         if number_of_string_constants expression > maxParameters then log 0. else
           substitute_string_constants stringConstants expression |> List.map ~f:(fun p ->
               let p' = analyze_lazy_evaluation p in
               (* Returns loss *)
               let rec loop = function
                 | [] -> true
                 | (xs,y) :: e ->
                   try
                     (match run_for_interval
                             timeout
                             (fun () -> run_lazy_analyzed_with_arguments p' xs = y)
                     with
                     | Some(true) -> loop e
                     | _ -> false)
                   with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                        | otherException -> begin
                            if otherException = EnumerationTimeout then raise EnumerationTimeout else false
                          end
               in
               let hit = loop examples in
               if hit
               then lc*.(Float.of_int (string_constants_length p))
               else log 0.) |> List.fold_right ~init:(log 0.) ~f:max)
  }


let keep_best_programs_in_frontier (k : int) (f : frontier) : frontier =
  {request = f.request;
   programs =  List.sort ~compare:(fun (_,a) (_,b) -> if a > b then -1 else 1) f.programs |> flip List.take k }

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

type hit_result = {hit_program: string;
                   hit_likelihood: float;
                   hit_prior: float;
                   hit_time: float;}

let enumerate_for_tasks (g: grammar) ?verbose:(verbose = true)
    ~maxFreeParameters
    ?budgetIncrement:(budgetIncrement = 1.)
    ?lowerBound:(lowerBound = 0.)
    ?upperBound:(upperBound = 99.)
    ?nc:(nc=1)
    ~timeout
    (* tasks and maximum frontier sizes *)
    (tf: (task*int) list)
  (* Returns, for each task, (program,logPrior,) *)
     : hit_result list list
  =

  set_enumeration_timeout timeout;

  let nt = List.length tf in
  let maximumFrontier = Array.of_list (tf |> List.map ~f:snd) in
  let tasks = Array.of_list (tf |> List.map ~f:fst) in

  let request = tasks.(0).task_type in
  assert (Array.for_all tasks ~f:(fun t -> t.task_type = request));

  (* Store the hits in a priority queue *)
  (* We will only ever maintain maximumFrontier best solutions *)
  let hits =
    Array.init nt ~f:(fun _ -> 
        Heap.create
          ~cmp:(fun h1 h2 ->
              let c = (h1.hit_likelihood+.h1.hit_prior) -. (h2.hit_likelihood+.h2.hit_prior) in
              if c < 0. then -1 else if c > 0. then 1 else 0) ()) in
  
  let lower_bound = ref lowerBound in

  let startTime = Time.now () in

  while not (enumeration_timed_out()) &&
          List.exists (range nt) ~f:(fun j -> Heap.length hits.(j) < maximumFrontier.(j))
       && !lower_bound +. budgetIncrement <= upperBound
    do
      let final_results =
        (* Returns a list of "final results" *)
        (* Each final result is [Array.map ~f:Heap.to_list hits] *)
        (* We flatten it to get a list of arrays of heaps *)
        enumerate_programs ~maxFreeParameters:maxFreeParameters ~nc:nc g request
          (!lower_bound) (!lower_bound +. budgetIncrement)
          ~final:(fun () -> [Array.map ~f:Heap.to_list hits])
          (fun p logPrior ->
             let mdl = 0.-.logPrior in

             assert( !lower_bound <= mdl);
             assert( mdl < budgetIncrement+.(!lower_bound));

             range nt |> List.iter ~f:(fun j -> 
                 let logLikelihood = tasks.(j).log_likelihood p in
                 if is_valid logLikelihood then begin
                   let dt = Time.abs_diff startTime (Time.now ())
                            |> Time.Span.to_sec in
                   Heap.add hits.(j)
                     {hit_program = string_of_program p;
                      hit_prior = logPrior;
                      hit_likelihood = logLikelihood;
                      hit_time = dt;} ;
                   while Heap.length hits.(j) > maximumFrontier.(j) do
                     Heap.remove_top hits.(j)
                   done;
                   if verbose then
                     Printf.eprintf
                       "\t(ocaml) HIT %s w/ %s\n" (tasks.(j).name) (string_of_program p)
                 end)) |> List.concat
      in

      if nc > 1 then
        (* merge the results from each of the parallel processes *)
        final_results |> List.iter ~f:(fun array_of_heaps ->
            range nt |> List.iter ~f:(fun j ->
                let new_heap = array_of_heaps.(j) in
                let old_heap = hits.(j) in
                List.iter new_heap ~f:(fun element ->
                    if not (Heap.mem old_heap ~equal:(=) element) then
                      (Heap.add old_heap element;
                       if Heap.length old_heap > maximumFrontier.(j)
                       then Heap.remove_top old_heap))))
      ;
      
      lower_bound := budgetIncrement+. (!lower_bound);

    done ;

    hits |> Array.to_list |> List.map ~f:Heap.to_list

