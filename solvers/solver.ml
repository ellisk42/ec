open Core

open Tower
(* open Vs *)
open Differentiation
open TikZ
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Task
open FastType

let load_problems channel =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel channel in
  let g = j |> member "DSL" in
  let logVariable = g |> member "logVariable" |> to_float in
  let productions = g |> member "productions" |> to_list |> List.map ~f:(fun p ->
    let source = p |> member "expression" |> to_string in
    let e = parse_program source |> safe_get_some ("Error parsing: "^source) in
    let t =
      try
        infer_program_type empty_context [] e |> snd
      with UnificationFailure -> raise (Failure ("Could not type "^source))
    in
    let logProbability = p |> member "logProbability" |> to_float in
    (e,t,logProbability,compile_unifier t))
  in
  (* Successfully parsed the grammar *)
  let g = {logVariable = logVariable; library = productions;} in

  let timeout = try
      j |> member "programTimeout" |> to_float
    with _ ->
      begin
        let defaultTimeout = 0.1 in
        Printf.eprintf
          "\t(ocaml) WARNING: programTimeout not set. Defaulting to %f.\n"
          defaultTimeout ;
        defaultTimeout
      end
  in

  (* Automatic differentiation parameters *)
  let differentiable =
    productions |> List.exists ~f:(fun (e,_,_,_) -> is_base_primitive e && "REAL" = primitive_name e)
  in
  let parameterPenalty =
    try j |> member "parameterPenalty" |> to_float
    with _ -> 0.
  in
  let maxParameters =
    try j |> member "maxParameters" |> to_int
    with _ -> 99
  in
  let lossThreshold =
    try Some(j |> member "lossThreshold" |> to_float)
    with _ -> None
  in

  (* string constant parameters *)
  let stringConstants =
    try j |> member "stringConstants" |> to_list |> List.map ~f:to_string
        |> List.map ~f:(String.to_list)
    with _ -> []
  in
  let is_constant_task = productions |> List.exists ~f:(fun (p,_,_,_) ->
      is_base_primitive p && primitive_name p = "STRING")
  in
  
  let guess_type elements =
    if List.length elements = 0 then t0 else
      
    let context = ref empty_context in
    let rec guess x =
      try ignore(x |> to_int); tint with _ ->
      try ignore(x |> to_float); treal with _ ->
      try ignore(x |> to_bool); tboolean with _ ->
      try
        let v = x |> to_string in
        if String.length v = 1 then tcharacter else tstring
      with _ ->
      try
        let l = x |> to_list in
        let (t,k) = makeTID !context in
        context := k;
        l |> List.iter ~f:(fun y ->
            let yt = guess y in
            context := unify !context yt t);
        tlist (applyContext !context t |> snd)
      with _ -> raise (Failure "Could not guess type")
    in
    let ts = elements |> List.map ~f:guess in
    let t0 = List.hd_exn ts in
    try
      ts |> List.iter ~f:(fun t -> context := (unify (!context) t0 t));
      applyContext !context t0 |> snd
    with UnificationFailure -> begin
        Printf.eprintf "Failure unifying types: %s\n"
          (ts |> List.map ~f:string_of_type |> join ~separator:"\t");
        assert false
      end
  in

  let rec unpack x =
    try magical (x |> to_int) with _ ->
    try magical (x |> to_float) with _ ->
    try magical (x |> to_bool) with _ ->
    try
      let v = x |> to_string in
      if String.length v = 1 then magical v.[0] else magical v
    with _ ->
    try
      x |> to_list |> List.map ~f:unpack |> magical
    with _ -> raise (Failure "could not unpack")
  in

  let tf = j |> member "tasks" |> to_list |> List.map ~f:(fun j -> 
      let e = j |> member "examples" |> to_list in
      let inputTypes = e |> List.map ~f:(fun ex -> ex |> member "inputs" |> to_list) |>
                       List.transpose |> safe_get_some "Not all examples have the same number of inputs." |>
                       List.map ~f:guess_type in
      let outputType = e |> List.map ~f:(fun ex -> ex |> member "output") |> guess_type in
      let task_type = List.fold_right ~f:(fun l r -> l @> r) ~init:outputType inputTypes in
      let examples = e |> List.map ~f:(fun ex -> (ex |> member "inputs" |> to_list |> List.map ~f:unpack,
                                                  ex |> member "output" |> unpack)) in
      let maximum_frontier = j |> member "maximumFrontier" |> to_int in
      let name = j |> member "name" |> to_string in
      (* towers *)
      let tower_stuff =
        try
          Some(tower_task ~perturbation:(j |> member "perturbation" |> to_float)
                 ~maximumStaircase:(j |> member "maximumStaircase" |> to_float)
                 ~maximumMass:(j |> member "maximumMass" |> to_float)
                 ~minimumLength:(j |> member "minimumLength" |> to_float)
                 ~minimumArea:(j |> member "minimumArea" |> to_float)
                 ~minimumHeight:(j |> member "minimumHeight" |> to_float)
                 ~stabilityThreshold:0.5)
        with _ -> None
      in

      let task_type = if is_some tower_stuff then ttower else task_type in
      
      let task = 
        (match tower_stuff with
         | Some(ts) -> ts
         | None -> 
           if differentiable
           then differentiable_task ~parameterPenalty:parameterPenalty
               ~lossThreshold:lossThreshold ~maxParameters:maxParameters
           else if is_constant_task then
             constant_task ~maxParameters:maxParameters ~parameterPenalty:parameterPenalty ~stringConstants:stringConstants else supervised_task)
          ~timeout:timeout name task_type examples
      in 
      (task, maximum_frontier))
  in

  let verbose = try j |> member "verbose" |> to_bool      
    with _ -> false
  in

  let lowerBound =
    try j |> member "lowerBound" |> to_float
    with _ -> 0.
  in

  let upperBound =
    try j |> member "upperBound" |> to_float
    with _ -> 99.
  in

  let budgetIncrement =
    try j |> member "budgetIncrement" |> to_float
    with _ -> 1.
  in

  let timeout = j |> member "timeout" |> to_float in
  let nc =
    try
      j |> member "nc" |> to_int 
    with _ -> 1
  in
  (tf,g,
   lowerBound,upperBound,budgetIncrement,
   nc,timeout,verbose)

let export_frontiers tf solutions : string =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let serialization : Yojson.Basic.json =
    `Assoc(List.map2_exn tf solutions ~f:(fun (t,_) ss ->
        (t.name, `List(ss |> List.map ~f:(fun s ->
             `Assoc([("program", `String(s.hit_program));
                     ("time", `Float(s.hit_time));
                     ("logLikelihood", `Float(s.hit_likelihood));
                     ("logPrior", `Float(s.hit_prior))]))))))
  in pretty_to_string serialization
;;

let main() =
  let (tf,g,
     lowerBound,upperBound,budgetIncrement,
     nc,timeout, verbose) =
    load_problems stdin in
  if List.exists tf ~f:(fun (t,_) -> t.task_type = ttower) then update_tower_cash() else ();
  let solutions =
    enumerate_for_tasks ~lowerBound:lowerBound ~upperBound:upperBound ~budgetIncrement:budgetIncrement
    ~verbose:verbose ~nc:nc ~timeout:timeout g tf
  in
  export_frontiers tf solutions |> print_string ;;

main();; 


(* test_best_enumeration();; *)
(* test_string();; *)
(* test_version();; *)

