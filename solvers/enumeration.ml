open Core

open Parallel

open Differentiation
open Utils
open Type
open Program
open Grammar

type frontier = {
  programs: (program*float) list;
  request: tp
}

let string_of_frontier f =
  join ~separator:"\n" ([Printf.sprintf "Frontier with request %s:" (string_of_type f.request)] @
                        (f.programs |> List.map ~f:(fun (p,l) -> Printf.sprintf "%f\t%s"
                                                       l (string_of_program p))))

let deserialize_frontier j =
  let open Yojson.Basic.Util in
  let request = j |> member "request" |> deserialize_type in
  let programs = j |> member "programs" |> to_list |> List.map ~f:(fun j ->
      (j |> member "program" |> to_string |> parse_program |> get_some |> strip_primitives,
       j |> member "logLikelihood" |> to_float))
  in
  {programs;request}

let serialize_frontier f =
  let open Yojson.Basic in
  let j : json =
    `Assoc(["request",serialize_type f.request;
            "programs",`List(f.programs |> List.map ~f:(fun (p,l) ->
                `Assoc(["program",`String(string_of_program p);
                       "logLikelihood",`Float(l)])))])
  in
  j

let violates_symmetry f a n = 
  if not (is_base_primitive f) then false else
    let a = application_function a in
    if not (is_base_primitive a) then false else 
      match (n, primitive_name f, primitive_name a) with
      (* McCarthy primitives *)
      | (0,"car","cons") -> true
      | (0,"car","empty") -> true
      | (0,"cdr","cons") -> true
      | (0,"cdr","empty") -> true
      | (_,"+","0") -> true
      | (1,"-","0") -> true
      | (0,"+","+") -> true
      | (0,"*","*") -> true
      | (_,"*","0") -> true
      | (_,"*","1") -> true
      | (0,"empty?","cons") -> true
      | (0,"empty?","empty") -> true
      | (0,"zero?","0") -> true
      | (0,"zero?","1") -> true
      | (0,"zero?","-1") -> true
      (* bootstrap target *)
      | (1,"map","empty") -> true
      | (_,"zip","empty") -> true
      | (0,"fold","empty") -> true
      | (1,"index","empty") -> true
      | (_,"left","left") -> true
      | (_,"left","right") -> true
      | (_,"right","right") -> true
      | (_,"right","left") -> true
      (** CLEVR primitives **)
      | (0,"clevr_car","clevr_add") -> true
      | (0,"clevr_car","clevr_empty") -> true
      | (0,"clevr_cdr","clevr_add") -> true
      | (0,"clevr_cdr","clevr_empty") -> true
      | (0,"clevr_empty?","clevr_add") -> true
      | (0,"clevr_empty?","clevr_empty") -> true
      | (1,"clevr_map","clevr_empty") -> true
      | (0,"clevr_fold","clevr_empty") -> true
      (* | (_,"tower_embed","tower_embed") -> true *)
      | _ -> false

(* For now this is disabled and is not used *)
let violates_commutative f x y =
    match f with
    | "eq?" | "+" -> compare_program x y > 0
    | _ -> false

(* Best first enumeration *)
let primitive_unknown t g = Primitive(t, "??", ref g |> magical);;
type turn =
  |A of tp
  |L
  |R


type best_first_state = {skeleton : program;
                         context : tContext;
                         path : turn list;
                         cost : float;
                         free_parameters : int;}

let path_environment p =
  p |> List.filter_map ~f:(function
      | A(t) -> Some(t)
      | _ -> None) |> List.rev

let string_of_state {skeleton;context;path;cost} =
  let string_of_turn = function
    | L -> "L"
    | R -> "R"
    | A(t) -> "["^string_of_type (applyContext context t |> snd)^"]"
  in
  Printf.sprintf "{cost=%f; skeleton=%s; path=%s; env=[%s];}"
    cost (string_of_program skeleton) (path |> List.map ~f:string_of_turn |> join ~separator:" ")
    (path |> path_environment |> List.map ~f:string_of_type |> join ~separator:",")

let state_finished {path;skeleton;} =
  match skeleton with
  | Primitive(_,"??",_) -> false
  | _ -> path = []

let initial_best_first_state request (g : grammar) =
  {skeleton = primitive_unknown request g;
   context = empty_context;
   path = [];
   cost = 0.0;
   free_parameters = 0;}

let rec follow_path e path =
  match (e,path) with
  | (Apply(f,_), L :: p') -> follow_path f p'
  | (Apply(_,x), R :: p') -> follow_path x p'
  | (Abstraction(body),(A(_)) :: p') ->
    follow_path body p'
  | (Primitive(t,"??",_), []) -> e
  | _ -> assert false

let rec modify_skeleton e q path =
  match (e,path) with
  | (Apply(f,x), L :: p') -> Apply(modify_skeleton f q p', x)
  | (Apply(f,x), R :: p') -> Apply(f, modify_skeleton x q p')
  | (Abstraction(body),(A(_)) :: p') ->
    Abstraction(modify_skeleton body q p')
  | (Primitive(t,"??",_), []) -> q
  | _ -> assert false

let unwind_path p =
  let rec unwind = function
    | [] -> []
    | (A(_)) :: r -> unwind r
    | R :: r -> unwind r
    | L :: r -> R :: r
  in 
  List.rev p |> unwind |> List.rev

let state_violates_symmetry {skeleton} =
  let rec r = function
    | Abstraction(b) -> r b
    | Apply(f,x) ->
      let (f,a) = application_parse (Apply(f,x)) in
      r f || List.exists a ~f:r ||
      List.existsi a ~f:(fun n x' -> violates_symmetry f x' n)
    | _ -> false
  in
  r skeleton
      

let state_successors ~maxFreeParameters cg state =
  let (request,g) = match follow_path state.skeleton state.path with
    | Primitive(t,"??",g) -> (t, !(g |> magical))
    | _ -> assert false
  in
  (* Printf.printf "request: %s\n" (string_of_type request); *)
  let context = state.context in
  let (context,request) = applyContext context request in

  match request with
  | TCon("->",[argument_type;return_type],_) ->
    [{skeleton = modify_skeleton state.skeleton (Abstraction(primitive_unknown return_type g)) state.path;
      path = state.path @ [A(argument_type)];
      free_parameters = state.free_parameters;
      context = context;
      cost = state.cost;}]
  | _ ->
    let environment = path_environment state.path in
    let candidates = unifying_expressions g environment request context in
    candidates |> List.map ~f:(fun (candidate, argument_types, context, ll) ->
        let new_free_parameters = number_of_free_parameters candidate in
        let argument_requests = match candidate with
          | Index(_) -> argument_types |> List.map ~f:(fun at -> (at,cg.variable_context))
          | _ ->  List.Assoc.find_exn cg.contextual_library candidate ~equal:program_equal |>
                  List.zip_exn argument_types
        in

        match argument_types with
        | [] -> (* terminal - unwind the recursion *)
          {context;
           cost = state.cost -. ll;
           free_parameters = state.free_parameters + new_free_parameters;
           path = unwind_path state.path;
           skeleton = modify_skeleton state.skeleton candidate state.path;}
        | first_argument :: later_arguments -> (* nonterminal *)
          let application_template =
            List.fold_left argument_requests ~init:candidate
              ~f:(fun e (a,at) -> Apply(e,primitive_unknown a at))
          in
          {skeleton = modify_skeleton state.skeleton application_template state.path;
           cost = state.cost -. ll;
           free_parameters = state.free_parameters + new_free_parameters;
           context;
           path = state.path @ List.map ~f:(fun _ -> L) later_arguments @ [R]; }) |>
    List.filter ~f:(fun new_state -> (not (state_violates_symmetry new_state)) &&
                                     new_state.free_parameters <= maxFreeParameters)

let best_first_enumeration ?lower_bound:(lower_bound=None)
    ?upper_bound:(upper_bound=None) ?frontier_size:(frontier_size=150) ~maxFreeParameters (cg : contextual_grammar) (request : tp) =
  let lower_bound = match lower_bound with
    | None -> -1.0
    | Some(lb) -> lb
  in
  let upper_bound = match upper_bound with
    | None -> 9999.0
    | Some(ub) -> ub
  in
  
  let completed = ref [] in
  
  let pq =
  Heap.create
      ~cmp:(fun s1 s2 ->
              let c = s1.cost -. s2.cost in
              if c < 0. then -1 else if c > 0. then 1 else 0) ()
  in
  Heap.add pq (initial_best_first_state request cg.no_context);

  while Heap.length pq > 0 && Heap.length pq < frontier_size do
    let best = Heap.pop_exn pq in
    assert (not (state_finished best));
    (* Printf.printf "\nParent:\n%s\n" (string_of_state best); *)
    state_successors ~maxFreeParameters:maxFreeParameters cg best |> List.iter ~f:(fun child ->
        if state_finished child
        then
          (if lower_bound <= child.cost && child.cost < upper_bound then completed := child :: !completed else ())
        else
          (if child.cost < upper_bound then Heap.add pq child else ()))
  done;

  (!completed,
   Heap.to_list pq)

      
(* Depth first enumeration *)
let enumeration_timeout = ref Float.max_value;;
let enumeration_timed_out() = Unix.time() > !enumeration_timeout;;
let set_enumeration_timeout dt =
  enumeration_timeout := Unix.time() +. dt;;


let rec enumerate_programs' (cg : contextual_grammar) (g: grammar) (context: tContext) (request: tp) (environment: tp list)
    (lower_bound: float) (upper_bound: float)
    ?maximumDepth:(maximumDepth = 9999)
    (* Symmetry breaking *)
    ?parent:(parent=None)
    (* We sometimes bound the number of free parameters *)
    ?maxFreeParameters:(maxFreeParameters=99)
    (callBack: program -> tContext -> float -> int -> unit) : unit =
  (* Enumerates programs satisfying: lowerBound <= MDL < upperBound *)
  (* INVARIANT: request always has the current context applied to it already *)
  if enumeration_timed_out() || maximumDepth < 1 || upper_bound < 0.0 then () else
    match request with
    | TCon("->",[argument_type;return_type],_) ->
      let newEnvironment = argument_type :: environment in
      enumerate_programs' ~maximumDepth:maximumDepth
        ~parent:None
        cg g context return_type newEnvironment
        lower_bound upper_bound
        ~maxFreeParameters:maxFreeParameters
        (fun body newContext ll fp -> callBack (Abstraction(body)) newContext ll fp)

    | _ -> (* not trying to enumerate functions *)
      let candidates = unifying_expressions g environment request context in
      candidates |>
      List.iter ~f:(fun (candidate, argument_types, context, ll) ->
          let mdl = 0.-.ll in
          if mdl >= upper_bound ||
             (match parent with
              | None -> false
              | Some((p,j)) -> violates_symmetry p candidate j)
          then () else
            let fp = number_of_free_parameters candidate in
            if fp > maxFreeParameters then () else
              let argument_requests = match candidate with
                | Index(_) -> argument_types |> List.map ~f:(fun at -> (at,cg.variable_context))
                | _ ->  List.Assoc.find_exn cg.contextual_library candidate ~equal:program_equal |>
                        List.zip_exn argument_types
              in
            enumerate_contextual_applications
              ~maximumDepth:(maximumDepth - 1)
              ~maxFreeParameters:(maxFreeParameters - fp)
              cg context environment
              argument_requests candidate
              (lower_bound+.ll) (upper_bound+.ll)
              (fun p k al fp' -> callBack p k (ll+.al) (fp + fp')))
and
  enumerate_contextual_applications
    ?maximumDepth:(maximumDepth = 9999)
    (cg : contextual_grammar)
    (context: tContext)  (environment: tp list)
    (argument_requests : (tp*grammar) list) (f: program)
    ?originalFunction:(originalFunction=f)
    ?argumentIndex:(argumentIndex=0)
    ?maxFreeParameters:(maxFreeParameters=99)
    (lower_bound: float) (upper_bound: float)
    (callBack: program -> tContext -> float -> int -> unit) : unit =
  (* Enumerates application chains satisfying: lowerBound <= MDL < upperBound *)
  (* returns the log likelihood of the arguments! not the log likelihood of the application! *)
  if enumeration_timed_out() || maximumDepth < 1 || upper_bound < 0.0 then () else
    match argument_requests with
    | [] -> (* not a function so we don't need any applications *)
      begin
        if lower_bound <= 0. && 0. < upper_bound then
          (* match f with
           * | Apply(Apply(Primitive(_,function_name,_),first_argument),second_argument)
           *   when violates_commutative function_name first_argument second_argument -> ()
           * | _ -> *) callBack f context 0.0 0
        else ()
      end
    | (first_argument, first_grammar)::later_arguments ->
      let (context,first_argument) = applyContext context first_argument in
      enumerate_programs' ~maximumDepth:maximumDepth ~maxFreeParameters:maxFreeParameters
        ~parent:(Some((originalFunction,argumentIndex)))
        cg first_grammar context first_argument environment
        0. upper_bound
        (fun a k ll fp ->
           let a = Apply(f,a) in
           enumerate_contextual_applications
             ~originalFunction:originalFunction
             ~argumentIndex:(argumentIndex+1)
             ~maximumDepth:maximumDepth
             ~maxFreeParameters:(maxFreeParameters - fp)
             cg k environment
             later_arguments a
             (lower_bound+.ll) (upper_bound+.ll)
             (fun a k a_ll fp' -> callBack a k (a_ll+.ll) (fp + fp')))

let dfs_around_skeleton cg ~maxFreeParameters ~lower_bound ~upper_bound state k =
  let rec free = function
    | Abstraction(b) -> free b
    | Apply(f,x) -> free f || free x
    | Index(_) -> false
    | Primitive(_,"??",_) -> true
    | Invented(_,_) -> false
    | Primitive(_,_,_) -> false
  in

  let rec parent_index = function
    (* Given that the input is an application, what is the identity of
       the function, and what is the index of the argument? *)
    | Apply(f,x) ->
      (match f with
       | Apply(_,_) ->
         let (f',n) = parent_index f in
         (f',n + 1)
       | _ -> (f,0))
    | _ -> assert false
  in

  let environment = path_environment state.path in

  let rec around e abstraction_depth ?parent:(parent=None) (* context abstraction_depth l u mfp k *) =
    match e with
    | Abstraction(body) ->
      let around_body = around ~parent:None body (1+abstraction_depth) in 
      fun context l u mfp k -> 
        around_body context l u mfp
          (fun body newContext ll fp -> k (Abstraction(body)) newContext ll fp)
    | Apply(f,x) when free x && (not (free f)) ->
      let around_argument = around ~parent:(Some(parent_index e)) x abstraction_depth in
      fun context l u mfp k ->       
        around_argument context l u mfp
          (fun x' newContext ll fp -> k (Apply(f,x')) newContext ll fp)
    | Apply(f,x) when free f && free x ->
      let around_argument = around ~parent:(Some(parent_index e)) x abstraction_depth in
      let around_function = around ~parent:None f abstraction_depth in
      fun context l u mfp k ->             
        around_function context 0. u mfp
          (fun f' context f_ll fp ->
             around_argument context (l+.f_ll) (u+.f_ll) (mfp - fp)
               (fun x' context x_ll fp' ->
                  k (Apply(f',x')) context (f_ll+.x_ll) (fp + fp')))
    | Apply(_,_) ->
      (* depth-first generation should ensure that functions are never free when their arguments are not free *)
      assert false
    | Primitive(t,"??",g) ->
      let g = !(g |> magical) in
      let environment = List.drop environment (List.length environment - abstraction_depth) in
      fun context l u mfp k ->             
        let (context, t) = applyContext context t in
        (* Printf.printf "Enumerating around type %s mfp = %d\n"
         *   (string_of_type t) (mfp); *)
        enumerate_programs' ~parent:parent cg g context t environment l u ~maxFreeParameters:mfp k
    | _ -> assert false
  in 
    
  around ~parent:None state.skeleton 0 state.context (lower_bound -. state.cost) (upper_bound -. state.cost)
    (maxFreeParameters - state.free_parameters)
    (fun e context ll _ -> k e context (ll-.state.cost))

let shatter_factor = ref 10;;

(* Putting depth first and best first gives us a parallel strategy for enumeration *)
let multicore_enumeration ?extraQuiet:(extraQuiet=false) ?final:(final=fun () -> []) ?cores:(cores=1) ?shatter:(shatter=None) cg request lb ub ~maxFreeParameters k =
  let shatter = match (shatter,cores) with
    | (Some(s),_) -> s
    | (None,1) -> 1
    | (None,c) -> !shatter_factor*c
  in
  
  let (finished, fringe) =
    best_first_enumeration ~lower_bound:(Some(lb)) ~upper_bound:(Some(ub)) ~frontier_size:shatter ~maxFreeParameters:maxFreeParameters cg request
  in

  if not extraQuiet then  
    (Printf.eprintf "\t(ocaml: %d CPUs. shatter: %d. |fringe| = %d. |finished| = %d.)\n"
    cores shatter (List.length fringe) (List.length finished);
     flush_everything());

  let strip_context p _ l = k p l in

  let continuation s =
    dfs_around_skeleton cg ~lower_bound:lb ~upper_bound:ub ~maxFreeParameters:maxFreeParameters s strip_context
  in

  let fringe = fringe |>
               List.sort ~compare:(fun s1 s2 ->
                   let d = s1.cost -. s2.cost in
                   if d > 0. then 1 else if d < 0. then -1 else 0)
  in

  if cores > 1 then
    let actions = fringe |>
                  List.map ~f:(fun s () -> continuation s) in
    let fringe_results = parallel_work ~nc:cores ~chunk:1 ~final:final actions in
    (* ignore(time_it "Evaluated finished programs" (fun () -> *)
    finished |> List.iter ~f:(fun s -> k s.skeleton (0.-.s.cost));
    final() :: fringe_results
  else begin 
    List.iter ~f:continuation fringe;
    finished |> List.iter ~f:(fun s -> k s.skeleton (0.-.s.cost));
    [final()]
  end
;;


let enumerate_programs ?extraQuiet:(extraQuiet=false) ?maxFreeParameters:(maxFreeParameters=0) ?final:(final=fun () -> []) ?nc:(nc=1) cg request lb ub k =
  let number_of_arguments = arguments_of_type request |> List.length in
  let definitely_recursive = grammar_has_recursion number_of_arguments cg.no_context in

  (* Strip out the recursion operators because they only occur at the top level *)
  let strip_recursion g =
    {g with     
     library =
       g.library |>
       List.filter ~f:(fun (p,_,_,_) -> not (is_recursion_primitive p)) |>
       (* sort library by number of arguments so that it will tend to explore shorter things first *)
       List.sort ~compare:(fun (_,a,_,_) (_,b,_,_) -> List.length (arguments_of_type a) - List.length (arguments_of_type b)) }
  in 
  let g' = {no_context=strip_recursion cg.no_context;
            variable_context=strip_recursion cg.variable_context;
            contextual_library=cg.contextual_library |> List.filter_map ~f:(fun (program, grammars) ->
                if is_recursion_primitive program then None else Some(
                    (program, grammars |> List.map ~f:strip_recursion)))} in

  let request' =
    if definitely_recursive then request @> request else request
  in

  let k' =
    if definitely_recursive then begin 
      fun p l ->
        let p' = 
          match p with
          | Abstraction(body) ->
            if variable_is_bound ~height:0 body then (* Used the fix point operator *)
              match number_of_arguments with
              | 1 ->
                Abstraction(Apply(Apply(primitive_recursion, Index(0)),p))
              | 2 -> Abstraction(Abstraction(Apply(Apply(Apply(primitive_recursion2, Index(1)),Index(0)),p)))
              | _ -> raise (Failure "number_of_arguments not supported by fix point")
            else body (* Remove unused recursion *)
          | _ -> raise (Failure "enumerated recursive definition that does not start with a lambda")
        in k p' l
    end
    else k
  in
  
  multicore_enumeration ~extraQuiet ~maxFreeParameters:maxFreeParameters ~final:final ~cores:nc g' request' lb ub k'



let test_recursive_enumeration () =
  let g = primitive_grammar [primitive_cons;primitive_car;primitive_cdr;primitive_is_empty;
                             primitive_empty;
                             primitive0;
                             primitive_recursion;primitive_recursion2;] in
  let request = (tlist tint @> tint @> tlist tint) in
  enumerate_programs (g |> make_dummy_contextual) request 0. 15.
    (fun p l ->
       Printf.printf "%s\t%f\n"
         (string_of_program p)
         l;
       flush_everything();
       let t = infer_program_type empty_context [] p |> snd in
       ignore(unify empty_context t request);
    Printf.printf "%s\n" (t |> string_of_type))
;;



let test_best_enumeration() = 
  let g = primitive_grammar [
      differentiable_placeholder;
      (* differentiable_zero; *)
      differentiable_one;
      differentiable_add;
      (* differentiable_division;
       * differentiable_power;
       * differentiable_subtract;
       * differentiable_multiply; *)
    ]
  in
  let mfp = 4 in 
  let request = treal @> treal in
  let frontier = ref [] in 
  let k p l = frontier := (string_of_program p, l) :: !frontier;
    assert (number_of_free_parameters p <= mfp);
    Printf.printf "%s\t%d\n" (string_of_program p) (number_of_free_parameters p)
  in
  let open Sys in
  enumerate_programs ~maxFreeParameters:mfp ~final:(fun () -> List.take !frontier 5) ~nc:(Sys.argv.(1) |> Int.of_string) (g |> make_dummy_contextual) request 0. (Sys.argv.(2) |> Float.of_string) k
;;
