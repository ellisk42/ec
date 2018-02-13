open Core.Std

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Task

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
      (e,t,logProbability))                        
  in
  (* Successfully parsed the grammar *)
  let g = {logVariable = logVariable; library = productions;} in

  let e = j |> member "examples" |> to_list in

  let guess_type elements =
    let rec guess context x =
      try ignore(x |> to_int); tint with _ ->
      try ignore(x |> to_bool); tboolean with _ ->
      try ignore(x |> to_string); tstring with _ ->
      try
        let l = x |> to_list in
        let (t,k) = makeTID !context in
        context := k;
        l |> List.iter ~f:(fun y ->
            let yt = guess context y in
            context := unify !context yt t);
        tlist (chaseType !context t |> fst)          
      with _ -> raise (Failure "Could not guess type")
    in
    let context = ref empty_context in
    let ts = elements |> List.map ~f:(guess context) in
    let t0 = List.hd_exn ts in
    ts |> List.iter ~f:(fun t -> context := (unify (!context) t0 t));
    chaseType !context t0 |> fst
  in
    
  let rec unpack x =
    try magical (x |> to_int) with _ ->
    try magical (x |> to_bool) with _ ->
    try magical (x |> to_string) with _ ->
    try
      x |> to_list |> List.map ~f:unpack |> magical
    with _ -> raise (Failure "could not unpack")
  in

  let inputType = e |> List.map ~f:(fun ex -> ex |> member "input") |> guess_type in
  let outputType = e |> List.map ~f:(fun ex -> ex |> member "output") |> guess_type in
  let task_type = inputType @> outputType in
  (* Printf.printf "Got task of type %s" (string_of_type task_type); *)
  (* print_newline (); *)
  let examples = e |> List.map ~f:(fun ex -> (ex |> member "input" |> unpack,
                                              ex |> member "output" |> unpack))
  in

  (* examples |> List.iter ~f:(fun (x,y) -> *)
  (*     Printf.printf "EXAMPLE\n"; *)
  (*     assert( task_type = (tlist tint @> tlist tint)); *)
  (*     let x : int list = magical y in *)
  (*     x |> List.iter ~f:(fun x -> Printf.printf "X\t%d\n" x)); *)
  (* assert false; *)
        

  let timeout = try
      j |> member "programTimeout" |> to_float
    with _ -> 0.1
  in

  (supervised_task ~timeout:timeout (j |> member "name" |> to_string) task_type examples,
   g,
   j |> member "solverTimeout" |> to_int,
   j |> member "maximumFrontier" |> to_int)

let export_frontier solutions : string =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let serialization : Yojson.Basic.json = 
  `List(solutions |> List.map ~f:(fun (p,lp,ll,t) ->
        `Assoc([("program", `String(string_of_program p));
                ("time", `Float(t));
                ("logLikelihood", `Float(ll));
                ("logPrior", `Float(lp))])))
  in pretty_to_string serialization

let main () =
  let (t,g,solverTimeout,maximumFrontier) = load_problem stdin in

  let solutions = enumerate_for_task ~verbose:false ~maximumFrontier:maximumFrontier ~timeout:solverTimeout g t in

  print_string (export_frontier solutions)
;;

  
main();;
