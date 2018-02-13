open Core.Std

open Utils
open Type
open Program
open Grammar

type frontier = {
  programs: (program*float) list;
  request: tp
}

let rec enumerate_programs (g: grammar) (context: tContext) (request: tp) (environment: tp list)
    (lower_bound: float) (upper_bound: float)
    (callBack: program -> tContext -> float -> unit) : unit =
  if upper_bound <= 0.0 then () else
    match request with
    | TCon("->",[argument_type;return_type]) ->
      let newEnvironment = argument_type :: environment in
      enumerate_programs g context return_type newEnvironment lower_bound upper_bound
      (fun body newContext ll -> callBack (Abstraction(body)) newContext ll)

    | _ -> (* not trying to enumerate functions *)
      unifying_expressions g environment request context |> 
      List.iter ~f:(fun (candidate, candidate_type, context, ll) ->
          let mdl = 0.-.ll in
          if mdl > upper_bound then () else 
            enumerate_applications g context candidate_type candidate environment
              (lower_bound+.ll) (upper_bound+.ll)
              (fun p k al -> callBack p k (ll+.al)))
and
  enumerate_applications (g: grammar) (context: tContext) (f_type: tp) (f: program) (environment: tp list) (lower_bound: float) (upper_bound: float) (callBack: program -> tContext -> float -> unit): unit =
  (* returns the log likelihood of the arguments! not the log likelihood of the application! *)
  if upper_bound <= 0. then () else 
    match arguments_and_return_of_type f_type with
    | ([], _) -> (* not a function so we don't need any applications *)
      if lower_bound < 0. && 0. <= upper_bound then callBack f context 0.0 else ()
    | (first_argument::_, _) ->
      enumerate_programs g context first_argument environment 0. upper_bound
        (fun a k ll ->
           let a = Apply(f,a) in
           let (applicationType,k) = chaseType k (right_of_arrow f_type) in
           enumerate_applications g k applicationType a environment (lower_bound+.ll) (upper_bound+.ll)
             (fun a k a_ll -> callBack a k (a_ll+.ll)))

(* let iterative_deepening_enumeration (g:grammar) (request:tp) (size:int) : frontier = *)
(*   let startTime = Time.now () in *)
(*   let rec deepen bound = *)
(*     let accumulator = ref [] in *)
(*     let _ = *)
(*       enumerate_programs g empty_context request [] bound *)
(*         (fun p _ ll -> accumulator := (p,ll) :: !accumulator) *)
(*     in *)
(*     let possibleSolutions = !accumulator in *)
(*     if List.length possibleSolutions<size then deepen (bound +. 1.0) *)
(*     else begin *)
(*       Printf.printf "Enumerated up to bound %f nats\n" bound; *)
(*       possibleSolutions *)
(*     end *)
(*   in *)
(*   let result = deepen 1.0 in *)
(*   Printf.printf "Enumerated %d programs of type %s in time %s\n" *)
(*     (List.length result) *)
(*     (string_of_type request) *)
(*     (Time.diff (Time.now ()) startTime |> Core.Span.to_string); *)
(*   {programs = result; request = request;} *)


