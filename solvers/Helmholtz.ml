open Core

open Program
open Enumeration
open Grammar
open Utils
open Timeout
open Type
open Tower
    
open Yojson.Basic

let helmholtz_enumeration (behavior_hash : program -> (int*json) option) ?nc:(nc=1) g request ~timeout ~maximumSize =
  assert (nc = 1); (* FIXME *)
  
  let behavior_to_programs = Hashtbl.Poly.create() in

  let update ~key ~data =
    let l,ps = data in
    match Hashtbl.find behavior_to_programs key with
    | None -> Hashtbl.set behavior_to_programs ~key ~data:data
    | Some((l',_)) when l' < l -> Hashtbl.set behavior_to_programs ~key ~data
    | Some((l',_)) when l' > l -> ()
    | Some((_,ps')) ->
      Hashtbl.set behavior_to_programs ~key ~data:(l, ps @ ps' |> List.dedup_and_sort ~compare:compare_program)
  in

  let merge other =
    Hashtbl.iteri other ~f:update
  in 

  set_enumeration_timeout timeout;

  let rec loop lb =
    if enumeration_timed_out() then () else begin 
      let final_results = 
        enumerate_programs ~nc:nc ~final:(fun () -> [behavior_to_programs])
          g request lb (lb+.1.5) (fun p l ->
              if Hashtbl.length behavior_to_programs > maximumSize then set_enumeration_timeout (-1.0) else
                match behavior_hash p with
                | Some(key) -> update ~key ~data:(l,[p])
                | None -> ()
            ) |> List.concat
      in
      if nc > 1 then final_results |> List.iter ~f:merge;
      loop (lb+.1.5)
    end
  in

  loop 0.;

  behavior_to_programs

let rec unpack x =
  let open Yojson.Basic.Util in
  
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

let rec pack t v : json =
  let open Yojson.Basic.Util in
  match t with
  | TCon("list",[t'],_) -> `List(magical v |> List.map ~f:(pack t'))
  | TCon("int",[],_) -> `Int(magical v)
  | TCon("bool",[],_) -> `Bool(magical v)
  | TCon("char",[],_) -> `String(magical v |> String.of_char)
  | _ -> assert false

let rec hash_json (j : json) : int =
  match j with
  | `List(xs) -> List.fold_right ~init:17 ~f:(fun x h -> Hashtbl.hash (h, hash_json x)) xs
  | `Int(n) -> Hashtbl.hash n
  | `String(x) -> Hashtbl.hash x
  | `Bool(x) -> Hashtbl.hash x
  | `Null -> Hashtbl.hash None
  | _ -> assert false

let special_helmholtz =   Hashtbl.Poly.create();;
let register_special_helmholtz name handle = Hashtbl.set special_helmholtz name handle;;


let default_hash ?timeout:(timeout=0.001) request inputs : program -> (int*json) option =
  let open Yojson.Basic.Util in

  (* convert json -> ocaml *)
  let inputs : 'a list list = unpack inputs in
  let return = return_of_type request in

  fun program ->
    let p = analyze_lazy_evaluation program in
    let outputs = inputs |> List.map ~f:(fun input ->
        try
          match run_for_interval timeout                  
                  (fun () -> run_lazy_analyzed_with_arguments p input)
          with
          | Some(value) -> Some(value |> pack return)            
          | _ -> None
        with (* We have to be a bit careful with exceptions if the
              * synthesized program generated an exception, then we just
              * terminate w/ false but if the enumeration timeout was
              * triggered during program evaluation, we need to pass the
              * exception on
             *)
        | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
        | _                   -> None) in
    if List.exists outputs ~f:is_some then
      let outputs = `List(outputs |> List.map ~f:(function
          | None -> `Null
          | Some(j) -> j)) in
      Some((hash_json outputs, outputs))
    else None

let string_hash ?timeout:(timeout=0.001) request inputs : program -> (int*json) option =
  let open Yojson.Basic.Util in

  (* convert json -> ocaml *)
  let inputs : 'a list list = unpack inputs in
  let return = return_of_type request in

  let testConstants=["x4";"a bc d"]  in
  let constants = testConstants |> List.map ~f:String.to_list in 

  fun program ->
    let constant_results = (* results from substituting with each constant *)
      constants |> List.map ~f:(fun constant ->
          match substitute_string_constants [constant] program with
          | [program'] -> 
            let p = analyze_lazy_evaluation program' in    
            let outputs = inputs |> List.map ~f:(fun input ->
                try
                  match run_for_interval timeout (fun () -> run_lazy_analyzed_with_arguments p input) with
                  | Some(value) -> Some(value |> pack return)
                  | _ -> None
                with
                | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                | _                   -> None) in
            if List.exists outputs ~f:is_some then
              let outputs = `List(outputs |> List.map ~f:(function
                  | None -> `Null
                  | Some(j) -> j)) in
              Some(outputs)
            else None
          | _ -> assert false)
    in
    if List.exists constant_results ~f:is_some then
      let outputs = `List(constant_results |> List.map ~f:(function
          | None -> `Null
          | Some(j) -> j)) in
      Some((hash_json outputs, outputs))
    else None
;;


register_special_helmholtz "string" string_hash;;

let tower_hash ?timeout:(timeout=0.001) request inputs : program -> (int*json) option =
  let open Yojson.Basic.Util in

  assert (request = (ttower @> ttower));

  fun program ->
    let arrangement = evaluate_discrete_tower_program timeout program in
    let l = List.length arrangement in
    let w = blocks_extent arrangement in
    let h = tower_height arrangement in
    if l = 0 || l > 100 || w > 360 || h > 250 then None else
      let j = `List(arrangement |> List.map ~f:(fun (a,b,c,d) -> `List([`Int(a);
                                                                        `Int(b);
                                                                        `Int(c);
                                                                        `Int(d);]))) in
      Some((hash_json j, j))
;;
register_special_helmholtz "tower" tower_hash;;

let logo_hash ?timeout:(timeout=0.001) request inputs : program -> (int*json) option =
  let open Yojson.Basic.Util in

  assert (request = (turtle @> turtle));

  let table = Hashtbl.Poly.create() in

  fun program ->
    let p = analyze_lazy_evaluation program in
    let l = run_for_interval timeout (fun () ->
        let x = run_lazy_analyzed_with_arguments p [] in
        let l = LogoLib.LogoInterpreter.turtle_to_list x in
        match Hashtbl.find table l with
        | Some(a) -> a
        | None -> begin
            let a = LogoLib.LogoInterpreter.turtle_to_array x 28 in
            Hashtbl.set table ~key:l ~data:a;
            a
          end)
    in
    match l with
    | None -> None
    | Some(a) ->
      let j = `List(range (28*28) |> List.map ~f:(fun i -> `Int(a.{i}))) in
      Some(((hash_json j, j)));;
register_special_helmholtz "LOGO" logo_hash;;


let run_job channel =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel channel in
  let request = j |> member "request" |> deserialize_type in
  let timeout = j |> member "timeout" |> to_float in
  let evaluationTimeout =
    try j |> member "evaluationTimeout" |> to_float
    with _ -> 0.001
  in
  let nc =
    try j |> member "CPUs" |> to_int
    with _ -> 1
  in
  let maximumSize =
    try j |> member "maximumSize" |> to_int
    with _ -> Int.max_value
  in
  let g = j |> member "DSL" in
  let g =
    try deserialize_grammar g |> make_dummy_contextual
    with _ -> deserialize_contextual_grammar g
  in

  let k =
    try Some(j |> member "special" |> to_string)
    with _ -> None
  in
  let k = match k with
    | None -> default_hash
    | Some(name) -> match Hashtbl.find special_helmholtz name with
      | Some(special) -> special
      | None -> (Printf.eprintf "Could not find special Helmholtz enumerator: %s\n" name; assert (false))
  in

  helmholtz_enumeration ~nc:nc (k ~timeout:evaluationTimeout request (j |> member "extras")) g request ~timeout ~maximumSize

let output_job result =
  let open Yojson.Basic.Util in
  let message : json = 
    `List(Hashtbl.to_alist result |> List.map ~f:(fun ((_, behavior), (l,ps)) ->
        `Assoc([(* "behavior", behavior; *)
                "ll", `Float(l);
                "programs", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_program)))])))
  in 
  pretty_to_string message

let _ =
  run_job Pervasives.stdin |> output_job |> print_string
