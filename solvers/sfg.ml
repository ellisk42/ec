open Core.Std

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Compression
open Differentiation


type sfg = {
  logVariables : variable list;
  productionProbabilities : (fragment*tp*variable) list list;
}

let string_of_categorized_grammar g =
  match (g.logVariables,g.productionProbabilities) with
  | (v0::vs, p0::ps) ->
    let inventory = p0 |> List.map ~f:(fun (f,_,_) -> f) in
    let s = "When starting a new program, the production probabilities are:\n"^
            "logVariable = "^(Float.to_string (variable_value v0))^"\n"^
            (p0 |> List.map ~f:(fun (f,t,l) -> Printf.sprintf "%f\t%s\t%s"
                                   (variable_value l)
                                   (string_of_type t)
                                   (string_of_fragment f)) |> join ~separator:"\n")^
            (List.zip_exn vs ps |> List.mapi ~f:(fun j (v,p) ->
                 "\nWhen starting an argument to fragment "^string_of_fragment (List.nth_exn inventory j)^
                 ", the production probabilities are:\n"^
                 "logVariable = "^(Float.to_string (variable_value v))^"\n"^
                 (p |> List.map ~f:(fun (f,t,l) -> Printf.sprintf "%f\t%s\t%s" (variable_value l) (string_of_type t) (string_of_fragment f)) |> join ~separator:"\n")^"\n") |> join ~separator:"\n")
                 
    in s
  | _ -> raise (Failure "string_of_categorized_grammar")

let categorized_of_fragment_grammar (f : fragment_grammar) =
  let number_of_productions = List.length f.fragments + 1 in
  {logVariables = (1--number_of_productions) |> List.map ~f:(fun _ -> random_variable ());
   productionProbabilities = (1--number_of_productions) |> List.map ~f:(fun _ ->
       f.fragments |> List.map ~f:(fun (f,t,_) -> (f,t,random_variable ())))}
  
let likelihood_under_sfg (g : sfg) (request : tp) (expression : program) : variable =
  (* Any chain of applications could be broken up at any point by a
     fragment. This enumerates all of the different ways of breaking up an
     application chain into a function and a list of arguments.
     Example: (+ 1 2) -> [(+,[1,2]), (+1,[2])] *)
  let rec possible_application_parses (p : program) : (program*(program list)) list =
    match p with
    | Apply(f,x) ->
      [(p,[])] @
      (possible_application_parses f |> List.map ~f:(fun (fp,xp) -> (fp,xp @ [x])))
    | _ -> [(p,[])]
  in

  let unifying_fragments (production : int) (environment : tp list) (request : tp) context
    : (int*fragment*tp*tContext*variable) list =
    let logVariable = List.nth_exn g.logVariables production in
    let fragments = List.nth_exn g.productionProbabilities production in
    let candidates =
      fragments @ List.mapi ~f:(fun j t -> (FIndex(j),t,logVariable)) environment |>
      List.filter_mapi ~f:(fun i (p,t,ll) ->
          try
            let (t,context) = if not (is_fragment_index p) then instantiate_type context t else (t,context) in
            let (_, return_type) = arguments_and_return_of_type t in
            let newContext = unify context return_type request in
            Some(i, p, t, newContext,ll)
          with UnificationFailure -> None)
    in
    let z = List.map ~f:(fun (_,_,_,_,ll) -> ll) candidates |> log_soft_max in
    (* IMPORTANT! We add one to the index *)
    (* This is so that the index correspondence to which subcategory the children will come from *)
    (* Subcategory 0 is the initial draw; the first subcategory is the draw from fragment 1; etc. *)
    List.map ~f:(fun (i,p,t,k,ll) -> (i+1,p,t,k,ll-&z)) candidates
  in

  
  let rec likelihood (context : tContext) (environment : tp list) (request : tp) (p : program) (production : int)
    : (tContext*variable) =
    let (request,context) = chaseType context request in
    match request with
    
    (* a function - must start out with a sequence of lambdas *)
    | TCon("->",[argument;return_type]) -> begin 
        let newEnvironment = argument :: environment in
        match p with
        | Abstraction(body) -> 
          likelihood context newEnvironment return_type body production 
        | _ -> (context, ~$ Float.neg_infinity)
      end
      
    | _ -> (* not a function so must be an application *)
      (* fragments we might match with based on their type *)
      let candidates = unifying_fragments production environment request context in
      
      (* The candidates are all different things that we could have possibly used *)
            
      (* For each way of carving up the program into a function and a list of arguments... *)
      possible_application_parses p |> List.map ~f:(fun (f,arguments) -> 
          List.map candidates ~f:(fun (candidate_index,candidate,unified_type,context,ll) ->
            try
              let (context, fragment_type, holes, bindings) = match f with
                | Index(i) ->
                  if FIndex(i) = candidate then (context, List.nth_exn environment i, [], FreeMap.empty)
                  else raise FragmentFail
                | _ -> 
                  bind_fragment context environment candidate f
              in
              (* Printf.printf "BOUND: %s & %s\n" (string_of_program f) (string_of_fragment candidate); *)

              (* The final return type of the fragment corresponds to the requested type *)
              let (context, fragment_request) =
                pad_type_with_arguments context (List.length arguments) request in
              let context = unify context fragment_request fragment_type in
              let (fragment_type, context) = chaseType context fragment_type in
              
              let (argument_types, _) = arguments_and_return_of_type fragment_type in
              if not (List.length argument_types = List.length arguments) then
                begin
                  Printf.printf "request: %s\n" (string_of_type request);
                  Printf.printf "program: %s\n" (string_of_program p);
                  Printf.printf "F = %s, xs = %s\n" (string_of_program f)
                    (arguments |> List.map ~f:string_of_program |> join ~separator:" ;; ");
                  Printf.printf "fragment: %s\n" (string_of_fragment candidate);
                  Printf.printf "fragment type: %s\n" (string_of_type fragment_type);
                  Printf.printf "argument types: %s\n" (argument_types |> List.map ~f:string_of_type |> join);
                  Printf.printf "arguments: %s\n" (arguments |> List.map ~f:string_of_program |> join);
                  assert false
                end
              else ()
              ;

              (* treat the holes and the bindings as though they were arguments *)
              let arguments = List.map holes ~f:(fun (_,h) -> h) @
                              List.map (FreeMap.to_alist bindings) ~f:(fun (_,(_,binding)) -> binding) @ 
                              arguments in
              let argument_types = List.map holes ~f:(fun (ht,_) -> ht) @
                                   List.map (FreeMap.to_alist bindings) ~f:(fun (_,(binding,_)) -> binding) @ 
                                   argument_types in

              let (application_likelihood, context) = 
                List.fold_right (List.zip_exn arguments argument_types)
                  ~init:(ll,context)
                  ~f:(fun (argument, argument_type) (ll,context) ->
                      let (context,argument_likelihood) =
                        likelihood context environment argument_type argument candidate_index
                      in (ll+&argument_likelihood, context))
              in
              (Some(context), application_likelihood)
            with | FragmentFail -> (None, ~$ Float.neg_infinity)
                 | UnificationFailure -> assert false)) |> List.concat |>

      (* Accumulate the probabilities from each parse. All of the contexts should be equivalent. *)
      List.fold_right ~init:(context, ~$ Float.neg_infinity) ~f:(fun (mayBeContext, ll)
                                                                   (oldContext, acc) ->
          match mayBeContext with
          | None -> begin
              assert (is_invalid (ll.data |> get_some));
              (oldContext,acc) end
          | Some(c) when is_valid (ll.data |> get_some) -> (c, log_soft_max [acc; ll])
          | Some(_) -> (oldContext, acc))
          
        
  in
  likelihood empty_context [] request expression 0 |> snd


let estimate_categorized_fragment_grammar (fg : fragment_grammar) (frontiers : frontier list) =
  let frontiers = frontiers |> List.filter ~f:(fun f -> List.length (f.programs) > 0) in
  let g = categorized_of_fragment_grammar fg in
  Printf.printf "%s\n" (string_of_categorized_grammar g);
  let joint = frontiers |> List.map ~f:(fun f ->
      f.programs |> List.map ~f:(fun (p,_) -> likelihood_under_sfg g (f.request) p) |> log_soft_max)
              |> fold1 (+&) in
  let parameters = g.logVariables @ (g.productionProbabilities |> List.map ~f:(List.map ~f:(fun (_,_,q) -> q))
                                    |> List.concat) in
  ignore(gradient_descent (~$0. -& joint) parameters);
  Printf.printf "%s\n" (string_of_categorized_grammar g);
