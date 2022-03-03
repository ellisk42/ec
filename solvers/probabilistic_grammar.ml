open Core

open Type
open Program
open Utils
open Grammar
open Enumeration

type rule_argument = {number_of_lambdas: int; nonterminal: int}
type probabilistic_production ={production_probability: float;
                                production_constructor: program;
                                production_arguments: rule_argument list
}

type probabilistic_grammar ={
  productions: (probabilistic_production Array.t) Array.t;
  start_symbol: int;
  number_of_arguments: int;
}

let show_probabilistic (g:probabilistic_grammar) : string =
  Array.to_list g.productions |> List.mapi ~f:(fun i p ->
      Array.to_list p |> List.map ~f:(fun {production_probability=lp;
                                           production_constructor=k;
                                           production_arguments=arguments} ->
                                       let arguments = arguments |> List.map ~f:(fun ar -> Printf.sprintf "%dx<%d>" (ar.number_of_lambdas) (ar.nonterminal))
                                                       |> join ~separator:" " in
          Printf.sprintf "<%d> ::= %s %s\t\tw.p. e^%f\n" i (string_of_program k) arguments lp)
      |> join ~separator:"\n")
  |> join ~separator:"\n"
;;

let deserialize_PCFG j =
  let open Yojson.Basic.Util in
  {
  productions=
  j |> member "rules" |> to_list |> List.map ~f:(fun r ->
      r |> to_list |> List.map ~f:(fun production ->
          {production_probability= production |> member "probability" |> to_float;
           production_constructor= production |> member "constructor" |> to_string |> parse_program |> safe_get_some ("Error parsing: PCFG constructor");
           production_arguments= production |> member "arguments" |> to_list |>
                                 List.map ~f:(fun ar -> {number_of_lambdas= ar |> member "n_lambda" |> to_int;
                                                         nonterminal= ar |> member "nt" |> to_int})
          }) |> Array.of_list) |> Array.of_list;
  start_symbol= j |> member "start_symbol" |> to_int;
  number_of_arguments= j |> member "number_of_arguments" |> to_int;
};;


let rec likelihood_PCFG (g : probabilistic_grammar) ?nt:(nt=g.start_symbol) (p : program) : float
  =
  let f, arguments = application_parse p in
  let arguments = arguments |> List.map ~f:recursively_get_abstraction_body in
  
  g.productions.(nt) |> Array.to_list |> List.filter_map
    ~f:(fun {production_probability;production_constructor;production_arguments} ->
        if program_equal production_constructor f then
          Some(List.fold2_exn production_arguments arguments ~init:production_probability
                 ~f:(fun likelihood {nonterminal;} argument ->
                     likelihood +. likelihood_PCFG g ~nt:nonterminal argument))
        else None) |>
  maximum_by ~cmp:(fun x y -> if x>y then 1 else -1)

let rec bounded_recursive_enumeration ~lower_bound ~upper_bound (g : probabilistic_grammar) ~nt (callback : program -> float -> unit) : unit =
  (* Enumerates programs satisfying: lowerBound <= MDL < upperBound *)
  if enumeration_timed_out() || upper_bound < 0.0 then () else
    g.productions.(nt) |>
    Array.iter ~f:(fun {production_probability=lp;
                        production_constructor=k;
                        production_arguments=arguments} ->
                    if (0.-.lp)>=upper_bound then () else
                      bounded_recursive_enumeration_application
                        ~lower_bound:(lower_bound+.lp) ~upper_bound:(upper_bound+.lp)
                        g k arguments
                        (fun p argument_likelihood -> callback p (argument_likelihood+.lp)))
and
  bounded_recursive_enumeration_application
    ~lower_bound ~upper_bound (g : probabilistic_grammar)
    (f : program)
    (arguments : rule_argument list)
    (callback : program -> float -> unit)
  =
  match arguments with
  | [] -> if lower_bound <= 0. && 0. < upper_bound then callback f 0. else ()
  | {number_of_lambdas;nonterminal}::later_arguments ->
    bounded_recursive_enumeration ~lower_bound:0. ~upper_bound:upper_bound g ~nt:nonterminal 
      (fun first_argument first_argument_likelihood ->
         let first_argument = wrap_with_abstractions number_of_lambdas first_argument in
         let f = Apply(f, first_argument) in
         bounded_recursive_enumeration_application
           ~lower_bound:(lower_bound+.first_argument_likelihood)
           ~upper_bound:(upper_bound+.first_argument_likelihood)
           g f later_arguments
           (fun finished_program further_application_likelihood ->
           callback finished_program (further_application_likelihood+.first_argument_likelihood)))
;;

let bounded_recursive_enumeration ~lower_bound ~upper_bound (g : probabilistic_grammar) (callback : program -> float -> unit) : unit =
  bounded_recursive_enumeration ~lower_bound ~upper_bound g ~nt:(g.start_symbol) callback


let bottom_up_enumeration ?factor:(factor=10) ~lower_bound ~upper_bound (g : probabilistic_grammar) (callback : program -> float -> unit) : unit =
  let factor = Float.of_int factor in 
  let scale f = Int.of_float (f*.factor +. 0.5) in
  let table =
    g.productions |> Array.map ~f:(fun  _-> Array.create (scale upper_bound + 1) None)
  in
  let productions =
    g.productions |> Array.map ~f:Array.to_list
  in 
  let rec populate symbol cost : program list =
    if cost<0 then [] else 
    match table.(symbol).(cost) with
    | Some(x) -> x
    | None ->
      (let x =
        productions.(symbol) |>
        List.concat_map ~f:(fun {production_probability;
                           production_constructor;
                           production_arguments;} ->
                       let c = cost - scale (0.-.production_probability) in
                       applications c production_constructor production_arguments)
      in
      table.(symbol).(cost) <- Some(x);
      x)
  and applications c f arguments =
    if c<0 then [] else
      match c, arguments with
      | 0, [] -> [f]
      | _, [] -> []
      | 0, (_::_) -> []
      | _, [{nonterminal=a1;number_of_lambdas=l1}] ->
        populate a1 c |> List.map ~f:(fun a1 ->
            let a1=wrap_with_abstractions l1 a1 in
            Apply(f, a1))
      | _, [{nonterminal=a1;number_of_lambdas=l1};{nonterminal=a2;number_of_lambdas=l2}] ->
        (0--c) |> List.concat_map ~f:(fun c1 ->
            populate a1 c1 |> List.concat_map ~f:(fun a1 ->
                let a1 = wrap_with_abstractions l1 a1 in 
                populate a2 (c-c1) |> List.map ~f:(fun a2 ->
                    let a2 = wrap_with_abstractions l2 a2 in 
                    Apply(Apply(f, a1), a2))))
      | _, [{nonterminal=a1;number_of_lambdas=l1};{nonterminal=a2;number_of_lambdas=l2};{nonterminal=a3;number_of_lambdas=l3}] ->
        (0--c) |> List.concat_map ~f:(fun c1 ->
            populate a1 c1 |> List.concat_map ~f:(fun a1 ->
                let a1 = wrap_with_abstractions l1 a1 in 
                (0--(c-c1)) |> List.concat_map ~f:(fun c2 ->
                    populate a2 c2 |> List.concat_map ~f:(fun a2 ->
                        let a2 = wrap_with_abstractions l2 a2 in 
                        populate a3 (c-c1-c2) |> List.map ~f:(fun a3 ->
                            let a3 = wrap_with_abstractions l3 a3 in 
                            Apply(Apply(Apply(f, a1), a2), a3))))))
      | _, [{nonterminal=a1;number_of_lambdas=l1};{nonterminal=a2;number_of_lambdas=l2};{nonterminal=a3;number_of_lambdas=l3};{nonterminal=a4;number_of_lambdas=l4}] ->
        (0--c) |> List.concat_map ~f:(fun c1 ->
            populate a1 c1 |> List.concat_map ~f:(fun a1 ->
                let a1 = wrap_with_abstractions l1 a1 in 
                (0--(c-c1)) |> List.concat_map ~f:(fun c2 ->
                    populate a2 c2 |> List.concat_map ~f:(fun a2 ->
                        let a2 = wrap_with_abstractions l2 a2 in 
                        (0--(c-c1-c2)) |> List.concat_map ~f:(fun c3 ->
                            populate a3 c3 |> List.concat_map ~f:(fun a3 ->
                                let a3 = wrap_with_abstractions l3 a3 in 
                                populate a4 (c-c1-c2-c3) |> List.map ~f:(fun a4 ->
                                    let a4 = wrap_with_abstractions l4 a4 in 
                                    Apply(Apply(Apply(Apply(f, a1), a2), a3), a4))))))))
  in 

  let rec loop c=
    if enumeration_timed_out() || c >= scale upper_bound then
      ()
    else begin
      Printf.eprintf "%d/%d\n"
        c (scale upper_bound); flush_everything();
      let l = (0. -. Float.of_int c /. factor) in 
      populate (g.start_symbol) c |>
      List.iter ~f:(fun p ->
          callback (wrap_with_abstractions (g.number_of_arguments) p) l);
      loop (c+1);
    end
  in
  loop (scale lower_bound)

(* let possible_concrete_types ?nesting:(nesting=3) (g:grammar) : tp list = *)
(*   let base_types = ref [] in *)
(*   let kinds = ref [] in *)
(*   let rec record_type = function *)
(*     | TID(_) -> () *)
(*     | TCon(k, [], _) -> *)
(*       base_types := (kind k []) :: !base_types *)
(*     | TCon("->", arguments, _) -> *)
(*       arguments |> List.iter ~f:record_type *)
(*     | TCon(k, arguments, _) -> begin *)
(*         kinds := (kind k (arguments|>List.mapi ~f:(fun index _ -> TID(index)))) :: !kinds; *)
(*         arguments |> List.iter ~f:record_type *)
(*       end *)
(*   in *)
(*   g.library |> List.iter ~f:(fun (_, t, _, _)-> record_type t); *)

(*   base_types := !base_types |> List.dedup ~compare:compare_tp; *)
(*   kinds := !kinds |> List.dedup ~compare:compare_tp; *)

(*   let rec types_of_size s : tp list = *)
(*     if s<=1 then !base_types else *)
(*       !kinds |> List.concat_map ~f:(function *)
(*           | TCon(k, [_], _) -> *)
(*             types_of_size (s-1) |> List.map ~f:(fun subtype -> kind k [subtype]) *)
(*           | TCon(k, [_; _], _) -> *)
(*             (1--(s-2)) |> List.concat_map ~f:(fun first_size -> *)
(*                 types_of_size first_size |> List.concat_map ~f:(fun first_type -> *)
(*                     types_of_size (s-1-first_size) |> List.map ~f:(fun second_type -> *)
(*                       kind k [first_type; second_type]))) *)
            
(*           | _ -> raise (Failure "kind arity >2")             *)
(*         ) *)
(*   in *)

(*   let possible_types = (1--nesting) |> List.concat_map ~f:types_of_size in *)

(*   possible_types *)
(* ;; *)

(* let ground_instantiations t legal_groundings = *)
(*   let t = canonical_type t in *)
(*   let number_of_variables = next_type_variable t in *)


(*   List.init number_of_variables (fun _ -> legal_groundings) |> *)
(*   cartesian_multi_product |> *)
(*   List.map ~f:(fun bindings -> *)
(*       let k = makeTIDs (List.length bindings) empty_context in *)
(*       bindings |> List.foldi ~init:k ~f:(fun i k t -> unify k t (TID(i)))) |> *)
(*   List.map ~f:(fun k -> applyContext k t |> snd) *)
(* ;; *)

(* let flatten_grammar ?nesting:(nesting=3) (g:grammar) : probabilistic_grammar = *)
(*   let possible_types = possible_concrete_types ~nesting g in *)

(*   let nonterminal_table = Hashtbl.Poly.create() in *)
(*   let index_of_context request context = *)
(*     match Hashtbl.find nonterminal_table (request, context) with *)
(*     | Some(i) -> i *)
(*     | None -> *)
(*       let new_index = Hashtbl.length nonterminal_table in  *)
(*       Hashtbl.set nonterminal_table ~key:(request, context) ~data:new_index; *)
(*       new_index *)
(*   in *)

(*   let show_nonterminal request environment = *)
(*     Printf.sprintf "%s[%s]" *)
(*       (string_of_type request) *)
(*       (environment |> List.map ~f:string_of_type |> join ~separator:", ") *)
(*   in  *)

(*   let rec make_rules request environment = *)
(*       g.library |> List.iter ~f:(fun (p, t, l, _) -> *)
(*         let arguments, return = arguments_and_return_of_type t in *)
(*         match *)
(*           try Some(unify (makeTIDs (next_type_variable t) empty_context) return request) with UnificationFailure -> None *)
(*         with *)
(*         | None -> () *)
(*         | Some(context) ->  *)
(*           let _, t = applyContext context t in *)
(*           ground_instantiations t possible_types |> *)
(*           List.iter ~f:(fun t -> *)
(*               let arguments, _ = arguments_and_return_of_type t in *)
(*               let argument_rules = arguments |> List.map ~f:(fun t -> *)
(*                 (return_of_type t, List.rev (arguments_of_type t) @ environment)) *)
(*               in  *)
(*               Printf.eprintf "%s ::= %s %s : %s\n" *)
(*                 (show_nonterminal request environment) *)
(*                 (string_of_program p) *)
(*                 (argument_rules |> List.map ~f:(fun (t, environment) -> *)
(*                    show_nonterminal t environment) |> join ~separator:" ") *)
(*                 (string_of_type t)) *)
(*       ) *)
(*   in *)
(*   possible_types |> List.iter ~f:(fun t -> Printf.eprintf "%s\n" (string_of_type t)); *)
(*   possible_types |> List.iter ~f:(fun request -> make_rules request []); *)
  
  
(*   {productions = Array.of_list []} *)
  
