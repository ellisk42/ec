open Core.Std

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar

module FreeMap = Map.Make(Int)

type fragment =
  | FIndex of int
  | FAbstraction of fragment
  | FApply of fragment*fragment
  | FPrimitive of tp * string
  | FInvented of tp * program
  | FVariable

let rec string_of_fragment = function
  | FIndex(j) -> "$" ^ string_of_int j
  | FAbstraction(body) ->
    "(lambda "^string_of_fragment body^")"
  | FApply(p,q) ->
    "("^string_of_fragment p^" "^string_of_fragment q^")"
  | FPrimitive(_,n) -> n
  | FInvented(_,i) -> "#"^string_of_program i
  | FVariable -> "??"
    
let is_fragment_index = function
  | FIndex(_) -> true
  | _ -> false

let rec fragment_size = function
  | FIndex(_) -> 1
  | FVariable -> 1
  | FPrimitive(_,_) -> 1
  | FInvented(_,_) -> 1
  | FAbstraction(b) -> 1 + fragment_size b
  | FApply(p,q) -> 1 + fragment_size p + fragment_size q

let infer_fragment_type ?context:(context = empty_context) ?environment:(environment = [])
    (f : fragment) : tp =
  let bindings : (tp FreeMap.t) ref = ref FreeMap.empty in
  let rec infer context environment = function
    | FIndex(j) when j < List.length environment ->
      let (t,context) = List.nth_exn environment j |> chaseType context in (context,t)
    | FIndex(j) -> begin 
      match FreeMap.find !bindings (j - List.length environment) with
      | Some(t) -> (context,t)
      | None ->
        let (t,context) = makeTID context in
        bindings := FreeMap.add !bindings ~key:(j - List.length environment) ~data:t;
        (context,t)
      end
    | FPrimitive(t,_) -> let (t,context) = instantiate_type context t in (context,t)
    | FInvented(t,_) -> let (t,context) = instantiate_type context t in (context,t)
    | FVariable ->
      let (t,context) = makeTID context in (context,t)
    | FAbstraction(b) ->
      let (xt,context) = makeTID context in
      let (context,rt) = infer context (xt::environment) b in
      let (ft,context) = chaseType context (xt @> rt) in
      (context,ft)
    | FApply(f,x) ->
      let (rt,context) = makeTID context in
      let (context, xt) = infer context environment x in
      let (context, ft) = infer context environment f in
      let context = unify context ft (xt @> rt) in
      let (rt, context) = chaseType context rt in
      (context, rt)
  in
  let (context,t) = infer context environment f
  in chaseType context t |> fst

type fragment_grammar = {
  logVariable : float;
  fragments : (fragment*tp*float) list;
}

let string_of_fragment_grammar g =
  "logVariable = "^(Float.to_string g.logVariable)^"\n"^
  (join ~separator:"\n" (g.fragments |> List.map ~f:(fun (f,t,l) ->
     Float.to_string l^"\t"^string_of_type t^"\t"^string_of_fragment f)))

exception FragmentFail

(* Tries binding the program with the fragment. The type returned is
   that of the *fragment*, not that of the program or the program
   unified with the fragment *)
let rec bind_fragment context environment
    (f : fragment) (p : program) : tContext*tp*((tp*program) list)*((tp*program) FreeMap.t) =
  
  let combine context (l,m) (lp,mp) =
    let (context,free) = 
     FreeMap.fold2 m mp ~init:(context, FreeMap.empty)
       ~f:(fun ~key:key ~data:merge (context,accumulator) ->
         match merge with
         | `Both((t1,p1),(t2,p2)) when not (p1 <> p2) -> raise FragmentFail
         | `Both((t1,p1),(t2,p2)) -> begin 
             try
               let context = unify context t1 t2 in
               let (t,context) = chaseType context t1 in
               (context, FreeMap.add accumulator ~key:key ~data:(t,p1))
             with _ -> raise FragmentFail
           end
         | `Left(program_and_type) ->
           (context, FreeMap.add accumulator ~key:key ~data:program_and_type)
         | `Right(program_and_type) ->
           (context, FreeMap.add accumulator ~key:key ~data:program_and_type))
    in (context, l@lp, free)
    
  in
  
  let rec hoist ?d:(d = 0) j p = match p with
    | Primitive(_,_) -> p
    | Invented(_,_) -> p
    | Apply(f,x) -> Apply(hoist j f ~d:d, hoist j x ~d:d)
    | Abstraction(b) -> Abstraction(hoist j b ~d:(d+1))
    | Index(i) when i < d -> p (* bound within the hoisted code *)
    | Index(i) when i >= d+j -> Index(i - j) (* bound outside the hoisted code but also outside the fragment *)
    | Index(_) -> raise FragmentFail (* bound inside of the fragment and so cannot be hoisted *)
  in
  match (f,p) with
  | (FApply(a,b),Apply(m,n)) ->
    let (context, ft,holes1,free1) = bind_fragment context environment a m in
    let (context, xt,holes2,free2) = bind_fragment context environment b n in
    let (context, holes, free) = combine context (holes1,free1) (holes2,free2) in
    let (alpha, context) = makeTID context in
    let context =
      try
        unify context (xt @> alpha) ft
      with  _ -> raise FragmentFail
    in
    let (alpha, context) = chaseType context alpha in
    (context, alpha, holes, free)
    
  | (FPrimitive(t,n1),Primitive(_,n2)) when n1 = n2 ->
    let (t,context) = instantiate_type context t in
    (context,t,[],FreeMap.empty)
  | (FInvented(t,n1),Invented(_,n2)) when n1 = n2 ->
    let (t,context) = instantiate_type context t in
    (context,t,[],FreeMap.empty)
  | (FAbstraction(m),Abstraction(n)) ->
    let (alpha, context) = makeTID context in
    let (context,beta,holes,free) = bind_fragment context (alpha::environment) m n in
    (context, alpha @> beta, holes, free)
  | (FVariable, _) ->
    let (alpha, context) = makeTID context in
    let p = hoist (List.length environment) p in
    (context, alpha, [(alpha, p)], FreeMap.empty)
  | (FIndex(j),_) when j >= List.length environment ->
    let p = hoist (List.length environment) p in
    let (alpha, context) = makeTID context in
    (context, alpha, [],
     FreeMap.singleton (j - List.length environment) (alpha,p))
  | (FIndex(j),Index(k)) when j < List.length environment && j = k ->
    (context, List.nth_exn environment j, [], FreeMap.empty)
  | _ -> raise FragmentFail

let unifying_fragment_variables g environment request context : (fragment*tp*float) list = 
  (* construct variables and remove those that do not have the correct type *)
  let candidates = environment |> List.mapi ~f:(fun j t -> (FIndex(j),t,g.logVariable))
                   |> List.filter ~f:(fun (_,t,_) ->
                       try ignore(unify context t request); true
                       with UnificationFailure -> false)
  in
  (* normalize variables *)
  let log_number_of_variables = List.length candidates |> Float.of_int |> log in
  candidates |> List.map ~f:(fun (e,t,ll) ->
      (e,t,ll-.log_number_of_variables))
    
let unifying_fragments (g : fragment_grammar) (environment : tp list) (request : tp) (context : tContext)
   : (int*fragment*tp*tContext*float) list =
  let candidates = 
  g.fragments @ unifying_fragment_variables g environment request context |>
  List.filter_mapi ~f:(fun i (p,t,ll) ->
      try
        let (t,context) = if not (is_fragment_index p) then instantiate_type context t else (t,context) in
        let (_, return_type) = arguments_and_return_of_type t in
        let newContext = unify context return_type request in
        Some(i, p, t, newContext,ll)
      with UnificationFailure -> None)
  in
  let z = List.map ~f:(fun (_,_,_,_,ll) -> ll) candidates |> lse_list in
  List.map ~f:(fun (i,p,t,k,ll) -> (i,p,t,k,ll-.z)) candidates

type use_counter = {possible_uses: float list; actual_uses: float list;
                   possible_variables: float; actual_variables: float}

let use_plus u1 u2 = {possible_uses = u1.possible_uses +| u2.possible_uses;
                      actual_uses = u1.actual_uses +| u2.actual_uses;
                      possible_variables = u1.possible_variables +. u2.possible_variables;
                      actual_variables = u1.actual_variables +. u2.actual_variables;}

let scale_uses a u = {possible_uses = a *| u.possible_uses;
                      actual_uses = a *| u.actual_uses;
                      possible_variables = a *. u.possible_variables;
                      actual_variables = a *. u.actual_variables;}
                     

let zero_uses g =
  let n = List.length (g.fragments) in
  let u = zeros n in
  {possible_uses = u; actual_uses = u;
   possible_variables = 0.; actual_variables = 0.;}

let pseudo_uses a g =
  let n = List.length (g.fragments) in
  let u = replicate n a in
  {possible_uses = u; actual_uses = u;
   possible_variables = a; actual_variables = a;}
  
let one_hot_uses g j =
  let n = List.length (g.fragments) in
  assert (j < n);
  let u = zeros j @ [1.] @ zeros (n - j - 1) in
  {possible_uses = u; actual_uses = u;
   possible_variables = 0.; actual_variables = 0.;}

let one_hot_variable_use g =
  let n = List.length (g.fragments) in
  let u = zeros n in
  {possible_uses = u; actual_uses = u;
  possible_variables = 1.; actual_variables = 1.;}

let no_uses = {possible_uses = []; actual_uses = [];
               possible_variables = 0.; actual_variables = 0.;}

let show_uses g u =
  List.iter2_exn (g.fragments) (List.zip_exn (u.actual_uses) (u.possible_uses)) ~f:(fun (f,_,_) (a, p) ->
      (*       assert (a >= p); *)
      if a > 0. then Printf.printf "Fragment %s used %f/%f times\n" (string_of_fragment f) a p else ())

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

let test_possible_application_parses () =
  let test_cases = ["(+ k1 k2)"; "(* (+ k1 k2) k9)"; "(lambda $0)"; "k0"] in
  test_cases |> List.iter ~f:(fun e ->
      Printf.printf "Test case: %s\n" e;
      let e = parse_program e |> get_some in
      possible_application_parses e |> List.iter ~f:(fun (f,xs) ->
          Printf.printf "f = %s\txs = %s\n" (string_of_program f)
            (xs |> List.map ~f:string_of_program |> join ~separator:" ;; ")))


let likelihood_under_fragments (g : fragment_grammar) (request : tp) (expression : program) : float*use_counter =

 
  let rec likelihood (context : tContext) (environment : tp list) (request : tp) (p : program)
    : (tContext*float*use_counter) =
    let (request,context) = chaseType context request in
    match request with
    
    (* a function - must start out with a sequence of lambdas *)
    | TCon("->",[argument;return_type]) -> begin 
        let newEnvironment = argument :: environment in
        match p with
        | Abstraction(body) -> 
          likelihood context newEnvironment return_type body
        | _ ->
          Printf.printf "WARNING: likelihood: expected lambda for type %s, got %s\n"
            (string_of_type request) (string_of_program p);
          (context, Float.neg_infinity, zero_uses g)
      end
      
    | _ -> (* not a function so must be an application *)
      (* fragments we might match with based on their type *)
      let candidates = unifying_fragments g environment request context in
      
      (* The candidates are all different things that we could have possibly used *)
      let number_of_productions = List.length g.fragments in
      let used_indexes = candidates |> List.filter_map ~f:(fun (index,_,_,_,_) ->
          if index < number_of_productions then Some(index) else None) in
      let possible_uses = {actual_uses = zeros number_of_productions;
                           actual_variables = 0.;
                           possible_variables =
                             candidates |> List.exists ~f:(fun (_,f,_,_,_) -> is_fragment_index f) |>
                             float_of_bool;
                           possible_uses = 
                             (0--(number_of_productions - 1)) |> List.map ~f:(fun i ->
                                 if List.mem used_indexes i then 1.0 else 0.0)} in

      (* Printf.printf "For the requested type %s, the possible use of vector is:\n\t%s\n" *)
      (*   (string_of_type request) *)
      (*   (possible_uses.possible_uses |> List.map ~f:Float.to_string |> join ~separator:" "); *)
      
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

              let initial_use_vector = use_plus possible_uses (if is_index f
                                                               then one_hot_variable_use g
                                                               else one_hot_uses g candidate_index)
              in

              let (application_likelihood, context, application_uses) = 
                List.fold_right (List.zip_exn arguments argument_types)
                  ~init:(ll,context,initial_use_vector)
                  ~f:(fun (argument, argument_type) (ll,context,uv) ->
                      let (context,argument_likelihood,uvp) =
                        likelihood context environment argument_type argument
                      in (ll+.argument_likelihood, context, use_plus uv uvp))
              in
              (Some(context), application_likelihood, application_uses)
            with | FragmentFail -> (None, Float.neg_infinity, no_uses)
                 | UnificationFailure -> assert false)) |> List.concat |>

      (* Accumulate the probabilities from each parse. All of the contexts should be equivalent. *)
      List.fold_right ~init:(context, Float.neg_infinity, []) ~f:(fun (mayBeContext, ll, u)
                                                                   (oldContext, acc, us) ->
          match mayBeContext with
          | None -> begin
              assert (is_invalid ll);
              (oldContext,acc,us) end
          | Some(c) when is_valid ll -> (c, lse acc ll, (ll,u) :: us)
          | Some(_) -> (oldContext, acc, us))
      |> (fun (context,ll,listOfUses) ->
          let expectedUses = listOfUses |> List.map ~f:(fun (_ll,u) -> scale_uses (exp (_ll -. ll)) u) |>
                             List.fold_right ~init:(zero_uses g) ~f:use_plus in
          (context,ll,expectedUses))
          
        
  in
  let (_,ll,u) = likelihood empty_context [] request expression in
  (ll,u)


let rec fragment_of_program = function
  | Index(j) -> FIndex(j)
  | Abstraction(body) ->FAbstraction(fragment_of_program body)
  | Apply(p,q) -> FApply(fragment_of_program p, fragment_of_program q)
  | Primitive(t,n) -> FPrimitive(t,n)
  | Invented(t,i) -> FInvented(t,i)

let fragment_grammar_of_grammar (g : grammar) : fragment_grammar =
  {logVariable = g.logVariable;
  fragments = List.map (g.library) ~f:(fun (p,t,l) -> (fragment_of_program p, t, l))}

let close_fragment (f:fragment) : program =
  (* Mapping from <index beyond this fragment>, <which lambda it refers to, starting at 0> *)
  let mapping = Hashtbl.Poly.create () in
  let number_of_abstractions = ref 0 in
  let rec walk d = function
    | FVariable -> begin incr number_of_abstractions; Index(!number_of_abstractions + d - 1) end
    | FPrimitive(t,n) -> Primitive(t,n)
    | FInvented(t,n) -> Invented(t,n)
    | FAbstraction(b) -> Abstraction(walk (d+1) b)
    | FApply(f,x) -> Apply(walk d f, walk d x)
    | FIndex(j) when j < d -> Index(j)
    | FIndex(j) ->
      match Hashtbl.Poly.find mapping (j - d) with
      | Some(lambda_index) -> Index(lambda_index + d)
      | None -> begin
          incr number_of_abstractions;
          ignore(Hashtbl.Poly.add mapping ~key:(j - d) ~data:(!number_of_abstractions - 1));
          Index(!number_of_abstractions + d - 1)
        end
  in
  let body = walk 0 f in
  List.fold_left (0--(!number_of_abstractions - 1)) ~init:body ~f:(fun b _ -> Abstraction(b))

let rec fragments (d:int) (a:int) (p:program) =
  (* All of the fragments that have exactly a variables *)
  (* Assumed that we are in an environment with d lambda expressions *)
  let recursiveFragments = 
    match p with
    | Apply(f,x) ->
      List.map (0--a) ~f:(fun fa ->
          let functionFragments = fragments d fa f in
          let argumentFragments = fragments d (a-fa) x in
          List.cartesian_product functionFragments argumentFragments |>
          List.map ~f:(fun (fp,xp) -> FApply(fp,xp))) |>
      List.concat
    | Index(j) when a = 0 -> [FIndex(j)]
    | Index(_) -> []
    | Abstraction(body) ->
      fragments (d+1) a body |> List.map ~f:(fun b -> FAbstraction(b))
    | Primitive(t,n) when a = 0 -> [FPrimitive(t,n)]
    | Primitive(_,_) -> []
    | Invented(t,n) when a = 0 -> [FInvented(t,n)]
    | Invented(_,_) -> []
  in
  (* Given that there are d surrounding lambdas in the thing we are
     trying to fragment, and e surrounding lambdas in the fragmented
     self, are we allowed to replace it with a fragment?  The idea
     here is that any variable has to NOT refer to anything bound in
     the larger program which is not bound in the fragment*)
  let rec refers_to_bound_variables e = function
    | Apply(f,x) -> refers_to_bound_variables e f || refers_to_bound_variables e x
    | Abstraction(b) -> refers_to_bound_variables (e + 1) b
    | Index(j) -> j > e - 1 && (* has to not refer to something bound in the fragment *)
                  j < e + d (* has to not refer to something outside the whole program body *)
    | Primitive(_,_) -> false
    | Invented(_,_) -> false
  in
  if a = 1 && (not (refers_to_bound_variables 0 p)) then FVariable :: recursiveFragments else recursiveFragments

let is_fragment_nontrivial = function
  | FAbstraction(_) -> false
  | FApply(FAbstraction(_),FIndex(_)) -> false
  | FApply(FAbstraction(_),FVariable) -> false
  | f ->
    let rec variables height uses = function
      | FVariable -> 0
      | FAbstraction(b) -> variables (height + 1) uses b
      | FApply(f,x) -> variables height uses f + variables height uses x
      | FIndex(j) ->
        (match Hashtbl.find uses (j - height) with
         | None -> (ignore(Hashtbl.add uses ~key:(j - height) ~data:1); 0)
         | Some(old) -> 
           Hashtbl.set uses ~key:(j - height) ~data:(old + 1);
           if old = 1 then 2 else 1)
      | _ -> 0
    in
    let rec primitives = function
      | FPrimitive(_,_) -> 1
      | FInvented(_,_) -> 1
      | FAbstraction(b) -> primitives b
      | FApply(f,x) -> primitives f + primitives x
      | _ -> 0
    in
    let keep = Float.of_int (primitives f) +. (0.5 *. (Float.of_int (variables 0 (Int.Table.create()) f))) > 1.5 in
    (* if keep then Printf.printf "Keeping fragment %s\n" (string_of_fragment f) else *)
    (*   Printf.printf "Discarding fragment (%d,%d) %s\n" (primitives f) (variables f) (string_of_fragment f); *)
    keep

let rec propose_fragments (a:int) (program:program) : fragment list =
  let recursiveFragments =
    match program with
    | Apply(f,x) -> propose_fragments a f @ propose_fragments a x
    | Abstraction(b) -> propose_fragments a b
    | _ -> []
  in
  recursiveFragments @ (List.map (0--a) ~f:(fun ap -> fragments 0 ap program) |> List.concat
                        |> List.filter ~f:is_fragment_nontrivial)
   
let rec propose_fragments_from_frontiers (a:int) (frontiers: frontier list) : fragment list =
  let from_each = frontiers |> List.map ~f:(fun f ->
      f.programs |> List.map ~f:(fun (p,_) -> propose_fragments a p) |> List.concat |> remove_duplicates) in
  let counts = Hashtbl.Poly.create() in
  List.iter from_each ~f:(List.iter ~f:(fun f ->
      match Hashtbl.find counts f with
      | Some(k) -> Hashtbl.set counts ~key:f ~data:(k + 1)
      | None -> ignore(Hashtbl.add counts ~key:f ~data:1)));
  let thisa = Hashtbl.to_alist counts |> List.filter_map ~f:(fun (f,k) ->
      if k > 1 then Some(f) else None)
  in
  if a = 0 then thisa else thisa@(propose_fragments_from_frontiers (a - 1) frontiers)

let marginal_likelihood_of_frontiers (g : fragment_grammar) (frontiers : frontier list) : float =
  frontiers |> List.map ~f:(fun frontier ->
      frontier.programs |> List.map ~f:(fun (expression,_) -> 
          likelihood_under_fragments g frontier.request expression |> fst) |>
      lse_list) |> List.fold ~init:0.0 ~f:(+.)

let inside_outside ?alpha:(alpha = 1.0) (g : fragment_grammar) (frontiers : frontier list)
  : fragment_grammar =
  let compiled_uses = frontiers |> List.map ~f:(fun frontier ->
      let likelihoods = frontier.programs |> List.map ~f:(fun (expression,_) -> 
          likelihood_under_fragments g frontier.request expression)
      in
      let z = likelihoods |> List.map ~f:(fun (ll,_) -> ll) |> lse_list in
      likelihoods |> List.map ~f:(fun (ll,u) -> scale_uses (exp (ll-.z)) u)) |> List.concat 
  in
  let uses = compiled_uses |> List.fold_right ~init:(pseudo_uses alpha g) ~f:use_plus in
  let log_likelihoods = List.map2_exn (uses.actual_uses) (uses.possible_uses) ~f:(fun a p -> log a -. log p) in
  {logVariable = log uses.actual_variables -. log uses.possible_variables;
   fragments = List.map2_exn (g.fragments) log_likelihoods ~f:(fun (f,t,_) l -> (f,t,l))}

let induce_fragment_grammar ?lambda:(lambda = 2.) ?alpha:(alpha = 1.) ?beta:(beta = 1.)
    (candidates : fragment list) (frontiers : frontier list) (g0 : fragment_grammar)
     : fragment_grammar =
  let frontiers = frontiers |> List.filter ~f:(fun f -> List.length (f.programs) > 0) in
  (* types of the candidates *)
  let candidates = candidates |> List.map ~f:(fun f -> (f, infer_fragment_type f)) in

  let score (g : fragment_grammar) : fragment_grammar*float =
    (* flatten the fragment grammar's production probabilities *)
    let gp = {logVariable = -1.;
             fragments = g.fragments |> List.map ~f:(fun (f,t,_) -> (f,t,-1.))} in
    let gp = inside_outside ~alpha:alpha gp frontiers in
    let child_likelihood = marginal_likelihood_of_frontiers gp frontiers in
    let child_structure_penalty =
      lambda *. (List.fold_left gp.fragments ~init:0.0 ~f:(fun penalty (f,_,_) -> penalty +. (fragment_size f |> Float.of_int))) in
    let child_parameter_penalty = beta *. Float.of_int (List.length gp.fragments) in
    (gp, child_likelihood-.child_structure_penalty-.child_parameter_penalty)
  in
  
  let rec induce (previous_score : float) (count : int) (g : fragment_grammar) : fragment_grammar =
    if count = 0 then g else
      let unused_candidates = candidates |> List.filter ~f:(fun (f,_) ->
          not (List.exists (g.fragments)  ~f:(fun (fp,_,_) -> fp = f))) in
      let children = unused_candidates |> List.map ~f:(fun (f,ft) ->
          score {logVariable = g0.logVariable;
                 fragments = (f,ft,-1.)::g.fragments}) in
      let (best_child,best_score) = maximum_by ~cmp:(fun (_,l1) (_,l2) -> if l1 > l2 then 1 else -1) children in
      Printf.printf "Best new grammar (%f): %s\n" (best_score) (string_of_fragment_grammar best_child);
      flush stdout;
      if best_score > previous_score then
        induce best_score (count - 1) best_child
      else g

  in
  let (g0,s0) = score g0 in
  induce s0 100 g0

let grammar_of_fragment_grammar (g : fragment_grammar) : grammar =
  {logVariable = g.logVariable;
   library = g.fragments |> List.map ~f:(fun (f,_,l) ->
       let p = close_fragment f in
       let (_,t) = infer_program_type empty_context [] p in
       let p = if is_primitive p then p else Invented(t,p) in
       (p,t,l))}

let testClosing() =
  let p = Apply(Abstraction(Apply(Index(2),Index(0))),Index(1)) in
  [FApply(FIndex(0),FAbstraction(FIndex(99)))] @ propose_fragments 1 p |>
  List.map ~f:(fun f -> Printf.printf "%s\t%s\n" (string_of_fragment f) (close_fragment f |> string_of_program))

let testBinding (intended_outcome : bool) (f : fragment) (p : program) =
  Printf.printf "\nBinding test case: %s  ==  %s\n" (string_of_fragment f) (string_of_program p);
  try
    let (context,t,holes, bindings) = bind_fragment empty_context [] f p in
    (*   let context = unify context t (tint @> tint) in *)
    let (t,context) = chaseType context t in
    Printf.printf "Fragment return type: %s\n" (string_of_type t);
    FreeMap.iteri bindings ~f:(fun ~key ~data:(bound_type,bound_program) ->
        Printf.printf "Free variable binding of %d is %s : %s\n" key (string_of_program bound_program)
          (string_of_type @@ fst @@ chaseType context bound_type));
    holes |> List.iteri ~f:(fun h (t,v) ->
        Printf.printf "hole binding %d is %s : %s\n" h (string_of_program v) (string_of_type t));
    assert intended_outcome
  with _ -> begin
      Printf.printf "Failure binding %s with %s\n" (string_of_fragment f) (string_of_program p);
      assert (not intended_outcome)
    end
;;

let binding_test_cases () = 
  testBinding true
    (FApply(FIndex(0),FIndex(1))) (Apply(primitive "+" (tint @> tint @> tint) (+),
                                         primitive "k0" tint 0));

  testBinding true
    (FAbstraction(FApply(FIndex(0),FVariable)))
    (Abstraction(Apply(Index(0), primitive "k0" tint 0)))
  ;

  testBinding true
    (FAbstraction(FApply(FIndex(0),FVariable)))
    (Abstraction(Apply(Index(0), Index(99))))
  ;

  testBinding false
    (FAbstraction(FAbstraction(FApply(FIndex(1),FVariable))))
    (Abstraction(Abstraction(Apply(Index(1), Index(0)))))
  ;


  testBinding false
    (FAbstraction(FAbstraction(FApply(FIndex(1),FVariable))))
    ( Abstraction( Abstraction( Apply( Index(1),
                                       Abstraction(Apply(Index(0), Index(2)))))))
  ;

  testBinding true
    (FAbstraction(FAbstraction(FApply(FIndex(1),FVariable))))
    ( Abstraction( Abstraction( Apply( Index(1),
                                       Abstraction(Apply(Index(0), Index(3)))))))
;;

let application_parses_test_cases() =
  let rec possible_application_parses (p : program) : (program*(program list)) list =
    match p with
    | Apply(f,x) ->
      [(p,[])] @
      (possible_application_parses f |> List.map ~f:(fun (fp,xp) -> (fp,xp @ [x])))
    | _ -> [(p,[])]
  in

  possible_application_parses (Apply(Apply(Apply(Index(0),Index(1)),Index(2)),Index(3))) |> List.iter ~f:(fun (f,a) ->
      Printf.printf "%s w/arguments %s\n" (string_of_program f) (a |> List.map ~f:string_of_program |> join ~separator:" ;; "))
;;

(* application_parses_test_cases();; *)


(* binding_test_cases() *)
