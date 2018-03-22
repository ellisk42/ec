open Core

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
    
let load_problem channel =
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

  let e = j |> member "examples" |> to_list in

  let guess_type elements =
    let context = ref empty_context in
    let rec guess x =
      try
        let xs = x |> to_list in
        let l = List.hd xs |> get_some |> to_string in
        if l = "TRACESET" then ttrace else raise (Failure "Not a trace set")
      with _ ->
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
    ts |> List.iter ~f:(fun t -> context := (unify (!context) t0 t));
    applyContext !context t0 |> snd
  in
    
  let rec unpack x =
    try
      let xs = x |> to_list in
      let l = List.hd xs |> get_some |> to_string in
      if l = "TRACESET" then
        List.tl xs |> get_some |>
        List.map ~f:(fun command ->
            let command = command |> to_list in
            match command with
            | [c;x;y;] when "circle" = (to_string c) -> Circle(Vector(x |> to_int, y |> to_int))
            | [c;x1;y1;x2;y2;] when "rectangle" = (to_string c) ->
              Rectangle(Vector(x1 |> to_int, y1 |> to_int),
                        Vector(x2 |> to_int, y2 |> to_int))
            | _ -> raise (Failure "Trouble parsing trace set")) |> magical
      else raise (Failure "Not a trace set")
    with _ -> 
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

  let inputTypes = e |> List.map ~f:(fun ex -> ex |> member "inputs" |> to_list) |>
                   List.transpose |> safe_get_some "Not all examples have the same number of inputs." |> 
                   List.map ~f:guess_type in
  let outputType = e |> List.map ~f:(fun ex -> ex |> member "output") |> guess_type in
  let task_type = List.fold_right ~f:(fun l r -> l @> r) ~init:outputType inputTypes in
  (* Printf.printf "Got task of type %s" (string_of_type task_type); *)
  (* print_newline (); *)
  let examples = e |> List.map ~f:(fun ex -> (ex |> member "inputs" |> to_list |> List.map ~f:unpack,
                                              ex |> member "output" |> unpack))
  in

  let differentiable =
    productions |> List.exists ~f:(fun (e,_,_,_) -> is_base_primitive e && "REAL" = primitive_name e)
  in

  let timeout = try
      j |> member "programTimeout" |> to_float
    with _ -> begin 
        let defaultTimeout = 0.1 in
        Printf.eprintf "\t(ocaml) WARNING: programTimeout not set. Defaulting to %f.\n" defaultTimeout;
        defaultTimeout
      end
  in

  let verbose = try
      j |> member "verbose" |> to_bool
    with _ -> false
  in

  let solver_timeout = j |> member "solverTimeout" |> to_int in
  let maximum_frontier = j |> member "maximumFrontier" |> to_int in
  let name = j |> member "name" |> to_string in

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
  
  let t = if return_of_type task_type = ttrace then begin
      assert (List.length examples = 1);
      latex_task name (examples |> List.hd |> get_some |> snd |> magical)    
    end else
      (if differentiable
       then differentiable_task ~parameterPenalty:parameterPenalty
           ~lossThreshold:lossThreshold ~maxParameters:maxParameters
       else supervised_task)
        ~timeout:timeout name task_type examples
  in
  (t,g,
   lowerBound,upperBound,budgetIncrement,
   solver_timeout,maximum_frontier,verbose)

let export_frontier program_count solutions : string =
  (* solutions |> List.iter ~f:(fun (p,_,_,_) -> *)
  (*     Printf.eprintf "EXPORT: %s: %s. PROGRAM: %s\n" *)
  (*       (task.name) (task.task_type |> string_of_type) (string_of_program p)); *)
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let serialization : Yojson.Basic.json =
    `Assoc([("programCount", `Int(program_count));
            ("solutions", `List(solutions |> List.map ~f:(fun (p,lp,ll,t) ->
                 `Assoc([("program", `String(string_of_program p));
                         ("time", `Float(t));
                         ("logLikelihood", `Float(ll));
                         ("logPrior", `Float(lp))]))))])
  in pretty_to_string serialization

let main () =
  let (t,g,
       lowerBound,upperBound,budgetIncrement,solverTimeout,
       maximumFrontier,verbose) =
    load_problem stdin in

  let (solutions, program_count) =
    enumerate_for_task ~lowerBound:lowerBound ~upperBound:upperBound ~budgetIncrement:budgetIncrement
      ~verbose:verbose ~maximumFrontier:maximumFrontier ~timeout:solverTimeout g t
  in

  export_frontier program_count solutions |> print_string
;;

(* let test_filter() =
 *   let p = parse_program "(lambda (#(lambda (lambda (lambda (fix1 $0 (lambda (lambda (if (empty? $0) $3 ($4 (car $0) ($1 (cdr $0)))))))))) (lambda (lambda (if (empty? $1) $0 (cons $1 $0)))) empty $0))" |> get_some in
 *   let t = closed_inference p in
 *   Printf.printf "%s : %s\n" (string_of_program p) (string_of_type t);
 *   flush_everything();
 *   let a = analyze_lazy_evaluation p in
 *   let o : int list list = run_lazy_analyzed_with_arguments a [[[];[1;2;];[];[4;];[1;2;9]]] in
 *   o |> 
 *   List.map ~f:(join ~separator:";" % List.map ~f:Int.to_string) |> join ~separator:" || " |>
 *   Printf.printf "%s\n";
 *   flush_everything()
 * ;; *)


  
main();;


(* test_filter();; *)
