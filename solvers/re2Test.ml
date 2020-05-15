open Core
open Dreaming

open Program
open Enumeration
open Grammar
open Utils
open Timeout
open Type
open PolyValue
  
open Yojson.Basic

let test_program raw input = 
  let first_str = (List.hd_exn (List.hd_exn input)) in 
  let _ = Printf.eprintf "program: %s\n" raw in 
  let p = parse_program raw |> get_some in 
  let p = analyze_lazy_evaluation p in 
  let y = run_lazy_analyzed_with_arguments p (List.hd_exn input) in 
  Printf.eprintf "IN: %s OUT: %s \n" first_str y;;

let print_str_inputs unpacked_inputs request = 
  let return = Type.return_of_type request in
  let str_list = List.hd_exn unpacked_inputs in 
  str_list |> List.map ~f:(fun s -> 
    let _ = Printf.eprintf "IN: %s\n" s in 
    let poly_str = PolyValue.pack return s in 
    let constructed = PolyValue.to_string poly_str in 
    Printf.eprintf "Polystr: %s\n" constructed
    )

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
  let show_vars = 
    try j |> member "use_vars_in_tokenized" |> to_bool
    with _ -> false
  in
  let k =
    try Some(j |> member "special" |> to_string)
    with _ -> None
  in
  let k = match k with
    | None -> 
      Printf.eprintf "Using default hash\n";
      default_hash
    | Some(name) -> match Hashtbl.find special_helmholtz name with
      | Some(special) -> 
        Printf.eprintf "Found special Helmholtz enumerator for: %s\n" name; special
      | None -> (Printf.eprintf "Could not find special Helmholtz enumerator: %s\n" name; assert (false))
  in 
    (** Test individual enumeration **)
    (* let inputs = (j |> member "extras") in
    let unpacked_inputs : 'a list list = unpack inputs in
    
    (** Replace match **)
    let replace_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) _f $0))" in 
    let raw = "(lambda (_rflatten (map  "^ replace_te ^ "  (_rsplit (_rconcat _t _e) $0))))" in
  
    let vowel = "(_ror _u (_ror _o (_ror _i (_ror _a _e))))" in 
    let consonant = "(_rnot "^vowel ^")" in 
    let vowel_consonant = "(_rconcat " ^ vowel ^ " " ^ consonant ^" )" in 
    let replace_ae = "(lambda (if (_rmatch "^ vowel_consonant^" $0) _f $0))" in 
    let raw = "(lambda (_rflatten (map " ^ replace_ae ^  " (_rsplit "^vowel_consonant^" $0) ) ))" in
    
    test_program raw [["aeioudbcg"]] ;; *)
  
    (* helmholtz_enumeration ~nc:nc (k ~timeout:0.1 request (j |> member "extras")) g request ~timeout ~maximumSize *)
    let inputs = (j |> member "extras") in
    let unpacked_inputs : 'a list list = unpack inputs in
    let _ = print_str_inputs unpacked_inputs request in 
    let _ = print_str_inputs unpacked_inputs in 
    let behavior_hash = (k ~timeout:evaluationTimeout request (j |> member "extras")) in
    set_enumeration_timeout 1.0;
    let rec loop lb =
      if enumeration_timed_out() then () else begin 
        let final_results = 
          enumerate_programs ~extraQuiet:true ~nc:nc ~final:(fun () -> [])
            g request lb (lb+.1.5) (fun p l ->
                let _ = behavior_hash p in
                (Printf.eprintf "%s\n" (string_of_program p));
                
              ) |> List.concat
        in
        loop (lb+.1.5)
      end
    in
    loop 0.
    
let output_job ?maxExamples:(maxExamples=50000) ?show_vars:(show_vars=false) result =
  let open Yojson.Basic.Util in
  (* let result = Hashtbl.to_alist result in *)
  let results =
    let l = List.length result in
    if l < maxExamples then result else
      let p = (maxExamples |> Float.of_int)/.(l |> Float.of_int) in
      result |> List.filter ~f:(fun _ -> Random.float 1. < p)
  in
  let message : json = 
    `List(results |> List.map ~f:(fun (behavior, (l,ps)) ->
        `Assoc([(* "behavior", behavior; *)
                "ll", `Float(l);
                "programs", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_program)));  
                "tokens", `List(ps |> List.map ~f:(fun p -> `String(p |> string_of_tokens show_vars)));
                ])))
  in 
  message

let _ = 
  run_job Pervasives.stdin 
  (* |> remove_bad_dreams |> output_job |> to_channel Pervasives.stdout *)