open Core

open Utils
open Type
open Program
open Grammar

type frontier = {
  programs: (program*float) list;
  request: tp
}

type recursionEnumeration =
  | UnspecifiedRecursion
  | NoRecursion
  | Recursion of int

let rec enumerate_programs (g: grammar) (context: tContext) (request: tp) (environment: tp list)
    (lower_bound: float) (upper_bound: float)
    ?maximumDepth:(maximumDepth = 9999)
    ?recursion:(recursion = UnspecifiedRecursion)
    (callBack: program -> tContext -> float -> unit) : unit =

  (* Figure out whether we are allowed to recurse *)
  let recursion = match recursion with
    | UnspecifiedRecursion ->
      if grammar_has_recursion (arguments_of_type request |> List.length) g
      then begin
        (* Printf.printf "setting recursion to %d\n" (arguments_of_type request |> List.length); *)
        Recursion(arguments_of_type request |> List.length)
      end
      else NoRecursion
    | _ -> recursion
  in
  
  (* INVARIANT: request always has the current context applied to it already *)
  if maximumDepth < 1 || upper_bound <= 0.0 then () else
    match request with
    | TCon("->",[argument_type;return_type],_) ->
      let newEnvironment = argument_type :: environment in
      enumerate_programs ~maximumDepth:maximumDepth ~recursion:recursion
        g context return_type newEnvironment
        lower_bound upper_bound
        (fun body newContext ll -> callBack (Abstraction(body)) newContext ll)

    | _ -> (* not trying to enumerate functions *)
      let candidates = unifying_expressions g environment request context in
      let candidates =
        match recursion with
        | NoRecursion -> candidates |> List.filter ~f:(fun (p,_,_,_) -> not (is_recursion_primitive p))
        | Recursion(a) -> begin
            let filtered_candidates = 
              candidates |>
              List.filter ~f:(fun (p,_,_,_) -> not (is_recursion_primitive p) || (is_recursion_of_arity a p))
            in
            (* Printf.printf "Possibly recursive candidates: %s\n" *)
            (*   (filtered_candidates |> List.map ~f:(fun (p,_,_,_) -> string_of_program p) |> join ~separator:" "); *)
            filtered_candidates
          end
        | UnspecifiedRecursion -> assert false
      in
      candidates |> 
      List.iter ~f:(fun (candidate, candidate_type, context, ll) ->
          let mdl = 0.-.ll in
          if mdl > upper_bound then () else
            let argument_types = arguments_of_type candidate_type in
            enumerate_applications ~isRecursion:(is_recursion_primitive candidate)
              ~maximumDepth:(maximumDepth - 1)
              g context environment
              argument_types candidate 
              (lower_bound+.ll) (upper_bound+.ll)
              (fun p k al -> callBack p k (ll+.al)))
and
  enumerate_applications ?isRecursion:(isRecursion = false)
    ?maximumDepth:(maximumDepth = 9999)
    (g: grammar) (context: tContext)  (environment: tp list)
    (argument_types: tp list) (f: program)
    (lower_bound: float) (upper_bound: float)
    (callBack: program -> tContext -> float -> unit) : unit =
  (* returns the log likelihood of the arguments! not the log likelihood of the application! *)
  if maximumDepth < 1 || upper_bound <= 0. then () else 
    match argument_types with
    | [] -> (* not a function so we don't need any applications *)
      if lower_bound < 0. && 0. <= upper_bound then callBack f context 0.0 else ()
    | first_argument::later_arguments ->
      let first_argument = applyContext context first_argument in
      enumerate_programs ~maximumDepth:maximumDepth ~recursion:NoRecursion
        g context first_argument environment
        0. upper_bound
        (fun a k ll ->
           (* if isRecursion then *)
           (*   Printf.printf "enumerate_applications: is recursion. Target argument: %s\n" *)
           (*     (Index(List.length later_arguments - 1) |> string_of_program) *)
           (* else () *)
           (* ; *)
           (* Symmetry breaking w/ recursion *)
           if isRecursion && (not (program_equal a (Index(List.length later_arguments - 1)))) then () else 
             let a = Apply(f,a) in
             let recursionArgument = isRecursion && List.length later_arguments > 1 in
             enumerate_applications
               ~maximumDepth:(if recursionArgument then 1 else maximumDepth)
               ~isRecursion:recursionArgument
               g k environment
               later_arguments a
               (lower_bound+.ll) (upper_bound+.ll)
               (fun a k a_ll -> callBack a k (a_ll+.ll)))


let test_recursive_enumeration () =
  let g = primitive_grammar [primitive_cons;primitive_car;primitive_cdr;primitive_is_empty;
                             primitive_empty;
                             primitive_recursion;primitive_recursion2;] in
  enumerate_programs g empty_context (tlist tint @> tlist tint) [] 0. 10.
    (fun p _ l ->
       Printf.printf "%s\t%f\n" (string_of_program p) l)
;;

(* test_recursive_enumeration ();; *)
