open Core
open Unix

open CachingTable
open Timeout
open Utils
open Type
open Program
open Enumeration
open Grammar
open Differentiation
open Base.Exn
open Printexc

type task =
  { name: string; task_type: tp;
    log_likelihood: program -> float;
  }

(* let p2i : (LogoLib.LogoInterpreter.logo_instruction list,(int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t) Hashtbl.Poly.t = Hashtbl.Poly.create () *)
let p2i = CachingTable.create 100000


exception EnumerationTimeout

let gen_passwd length =
    let gen() = match Random.int(26+26+10) with
        n when n < 26 -> int_of_char 'a' + n
      | n when n < 26 + 26 -> int_of_char 'A' + n - 26
      | n -> int_of_char '0' + n - 26 - 26 in
    let gen _ = String.make 1 (char_of_int(gen())) in
    String.concat (Array.to_list (Array.init length gen))


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

let task_handler = Hashtbl.Poly.create();;
let register_special_task name handler = Hashtbl.set task_handler name handler;;

let recent_logo_program : (program*((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t option)) option ref = ref None;;
let run_recent_logo ~timeout program =
  match !recent_logo_program with
  | Some(program', bx) when program_equal program program' -> bx
  | _ ->
    let bx = run_for_interval timeout
                 (fun () ->
                    let p = analyze_lazy_evaluation program in
                    let x = run_lazy_analyzed_with_arguments p [] in
                    let l = LogoLib.LogoInterpreter.turtle_to_list x in
                    if not (LogoLib.LogoInterpreter.logo_contained_in_canvas l)
                    then None  
                    else match CachingTable.find p2i l with
                      | Some(bx) -> Some(bx)
                      | None -> 
                        let bx = LogoLib.LogoInterpreter.turtle_to_array x 28 in
                        CachingTable.set p2i l bx;
                        Some(bx))
    in
    let bx = match bx with
      | Some(Some(bx')) -> Some(bx')
      | Some(None) | None -> None
    in
    recent_logo_program := Some(program, bx);
    bx
;;

(** Sort and deduplicate the objects before comparing **)
register_special_task "clevrobjectlist" (fun extras
  ?timeout:(timeout = 0.05) name ty examples ->
  {
    name = name    ;
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
                      (fun () -> 
                        let output = run_lazy_analyzed_with_arguments p xs in
                        let compare = 
                        Program.compare_objs (magical output) (magical y)
                        in compare 
                        )
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
              | EnumerationTimeout  -> 
                let _ = Printf.eprintf("Enumeration timeout;") in 
                raise EnumerationTimeout
              | _                   ->  false
        in
        if loop examples
          then 0.0
          else log 0.0)
  }
);;

register_special_task "LOGO" (fun extras ?timeout:(timeout = 0.001) name ty examples ->
    let open Yojson.Basic.Util in
    let proto =
      try
        extras |> member "proto" |> to_bool
      with _ -> (Printf.eprintf "proto parameter not set! FATAL"; exit 1)                
    in

  let by = match examples with
      | [([0],y)] ->
          Bigarray.(Array1.of_array int8_unsigned c_layout (Array.of_list y))
      | _ -> failwith "not a turtle task" in
  { name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun p ->
        let s_inout =
          if proto then
            Some(
              open_connection
                (ADDR_UNIX("./prototypical-networks/protonet_socket"))
                )
          else None in

        let log_likelihood = (try begin
            match
              if true then
                (match run_recent_logo ~timeout p with
                 | Some(bx) when (LogoLib.LogoInterpreter.fp_equal bx by 0) -> Some(0.)
                 | _ -> None)
              else 
            run_for_interval
              timeout
              (fun () ->
                let x = run_lazy_analyzed_with_arguments (analyze_lazy_evaluation p) [] in
                let l = LogoLib.LogoInterpreter.turtle_to_list x in
                let bx =
                  match CachingTable.find p2i l with
                  | Some(x) -> x
                  | _ ->
                    let bx = LogoLib.LogoInterpreter.turtle_to_array x 28 in
                    CachingTable.set p2i l bx ;
                    bx
                in
                if proto then begin
                  let s_in, s_out = match s_inout with
                    | Some(x,y) -> x, y
                    | _ -> failwith "NOOOOO, don't dooo that !!!"
                  in
                  let bytes_version = Bytes.create (28 * 28) in
                  for i = 0 to (28 * 28) - 1 do
                    Bytes.set bytes_version i (char_of_int (bx.{i}))
                  done ;
                  let img = Bytes.to_string bytes_version in
                  output_binary_int s_out (String.length name) ;
                  output_string s_out name ;
                  output_binary_int s_out (String.length img) ;
                  output_string s_out img ;
                  flush s_out ;
                  let l = input_binary_int s_in in
                  float_of_string (really_input_string s_in l)
                end
                else begin
                  if (LogoLib.LogoInterpreter.fp_equal bx by 5) then 0.0
                  else log 0.0
                end)
          with
            | Some(x) -> x
            | _ -> log 0.0
        end
        with (* We have to be a bit careful with exceptions if the
              * synthesized program generated an exception, then we just
              * terminate w/ false but if the enumeration timeout was
              * triggered during program evaluation, we need to pass the
              * exception on
              *)
          | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
          | EnumerationTimeout  -> raise EnumerationTimeout
          | _                   -> log 0.0) in
        if proto then begin
          let s_in, s_out = match s_inout with
            | Some(x,y) -> x, y
            | _ -> failwith "NOOOOO, don't dooo that !!!"
          in
          output_binary_int s_out (String.length "DONE") ;
          output_string s_out "DONE" ;
          flush s_out ;
          shutdown_connection s_in ;
          close_in s_in ;
          (-. (100. *. log_likelihood))
        end else log_likelihood)
  });;



register_special_task "differentiable"
  (fun extras
    ?timeout:(timeout = 0.001) name ty examples ->


    let open Yojson.Basic.Util in
    let maybe_float name default =
      try
        extras |> member name |> to_float
      with _ -> default
    in
    let maybe_int name default =
      try
        extras |> member name |> to_int
      with _ -> default
    in 
    let temperature = maybe_float "temperature" 1. in
    let parameterPenalty = maybe_float "parameterPenalty" 0. in
    let maxParameters = maybe_int "maxParameters" 99 in
    let actualParameters = maybe_int "maxParameters" 99 in
    let restarts = maybe_int "restarts" 300 in
    let steps = maybe_int "steps" 50 in
    let lr = maybe_float "lr" 0.5 in
    let decay = maybe_float "decay" 0.5 in
    let grow = maybe_float "grow" 1.2 in
    let lossThreshold = try Some(extras |> member "lossThreshold" |> to_float) with _ -> None in
    let clipOutput = try Some(extras |> member "clipOutput" |> to_float) with _ -> None in
    let clipLoss = try Some(extras |> member "clipLoss" |> to_float) with _ -> None in
    let proportional = try
        extras |> member "proportional" |> to_bool
      with _ -> false
    in
    
                                         
  (* Process the examples and wrap them inside of placeholders *)
  let (argument_types, return_type) = arguments_and_return_of_type ty in
  let examples = examples |> List.map ~f:(fun (xs,y) ->
      (List.map2_exn argument_types xs ~f:placeholder_data,
      placeholder_data return_type y))
  in
    
  let loss = polymorphic_sse ~clipOutput ~clipLoss return_type in
  { name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun expression ->
         let (p,parameters) = replace_placeholders expression in
         assert (List.length parameters <= maxParameters);
        if List.length parameters > maxParameters || List.length parameters > actualParameters then log 0. else 
          let p = analyze_lazy_evaluation p in
          (* let predictions = examples |> List.map ~f:(fun (xs,_) -> *)
          (*     run_for_interval timeout (fun () -> run_lazy_analyzed_with_arguments p xs)) *)
          (* in *)
          (* if List.exists predictions ~f:is_none then 0. else *)
          (*   let predictions = predictions |> List.map ~f:get_some in *)
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
          let l = if proportional && List.length parameters > 0 then begin 
              assert (List.length parameters = 1);
              parameters |> List.iter ~f:(fun p -> update_variable p 1.);
              assert (false)
            end else 
                let l = l *& (~$ (1. /. n)) in
                let l = restarting_optimize (rprop ~lr ~decay ~grow)
                    ~attempts:restarts
                    ~update:0
                    ~iterations:(if List.length parameters = 0 then 0 else steps)
                    parameters l
                in l
          in
          match lossThreshold with
          | None -> 0. -. d*.parameterPenalty -. n *. l /. temperature
          | Some(t) ->
            if l < t then 0. -. d*.parameterPenalty else log 0.)
  });;


register_special_task "stringConstant" (fun extras
    (* ?parameterPenalty:(parameterPenalty=0.) *)
    (* ?maxParameters:(maxParameters=100) *)
    ?timeout:(timeout = 0.001)
    name ty examples ->
    let open Yojson.Basic.Util in
    let maybe_int name default =
      try
        extras |> member name |> to_int
      with _ -> default
    in 
    let stringConstants =
      extras |> member "stringConstants" |> to_list |> List.map ~f:to_string |> List.map ~f:(String.to_list)
    in
    let maxParameters = maybe_int "maxParameters" 99 in


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
  });;



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
                   hit_time: float;
                   hit_tokens: string;}

let enumerate_for_tasks (g: contextual_grammar) ?verbose:(verbose = true)
    ~maxFreeParameters
    ?budgetIncrement:(budgetIncrement = 1.)
    ?lowerBound:(lowerBound = 0.)
    ?upperBound:(upperBound = 99.)
    ?nc:(nc=1)
    ~timeout
    (* tasks and maximum frontier sizes *)
    (tf: (task*int) list)
  (* Returns, for each task, (program,logPrior) as well as the total number of enumerated programs *)
     : (hit_result list list)*int
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
              Float.compare (h1.hit_likelihood+.h1.hit_prior) (h2.hit_likelihood+.h2.hit_prior))
          ()) in
  
  let lower_bound = ref lowerBound in

  let startTime = Time.now () in

  let total_number_of_enumerated_programs = ref 0 in

  while not (enumeration_timed_out()) &&
          List.exists (range nt) ~f:(fun j -> Heap.length hits.(j) < maximumFrontier.(j))
       && !lower_bound +. budgetIncrement <= upperBound
  do
    let number_of_enumerated_programs = ref 0 in
      let final_results =
        (* Returns a list of "final results" *)
        (* Each final result is [Array.map ~f:Heap.to_list hits] *)
        (* We flatten it to get a list of arrays of heaps *)
        enumerate_programs ~maxFreeParameters:maxFreeParameters ~nc:nc g request
          (!lower_bound) (!lower_bound +. budgetIncrement)
          ~final:(fun () ->
              (* Printf.eprintf "%d\n" !number_of_enumerated_programs; flush_everything(); *)
              [(Array.map ~f:Heap.to_list hits, !number_of_enumerated_programs)])
          (fun p logPrior ->
             incr number_of_enumerated_programs;
             incr total_number_of_enumerated_programs;
             
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
                      hit_time = dt;
                      hit_tokens = string_of_tokens false p} ;
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
        final_results |> List.iter ~f:(fun (array_of_heaps, number_enumerated_here) ->
            total_number_of_enumerated_programs := !total_number_of_enumerated_programs +
                                                   number_enumerated_here;
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
    
  (hits |> Array.to_list |> List.map ~f:Heap.to_list,
   !total_number_of_enumerated_programs)

