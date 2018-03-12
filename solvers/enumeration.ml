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
  (* INVARIANT: request always has the current context applied to it already *)
  if upper_bound <= 0.0 then () else
    match request with
    | TCon("->",[argument_type;return_type],_) ->
      let newEnvironment = argument_type :: environment in
      enumerate_programs g context return_type newEnvironment lower_bound upper_bound
      (fun body newContext ll -> callBack (Abstraction(body)) newContext ll)

    | _ -> (* not trying to enumerate functions *)
      unifying_expressions g environment request context |> 
      List.iter ~f:(fun (candidate, candidate_type, context, ll) ->
          let mdl = 0.-.ll in
          if mdl > upper_bound then () else
            let argument_types = arguments_of_type candidate_type in
            enumerate_applications g context argument_types candidate environment
              (lower_bound+.ll) (upper_bound+.ll)
              (fun p k al -> callBack p k (ll+.al)))
and
  enumerate_applications (g: grammar) (context: tContext) (argument_types: tp list) (f: program) (environment: tp list) (lower_bound: float) (upper_bound: float) (callBack: program -> tContext -> float -> unit): unit =
  (* returns the log likelihood of the arguments! not the log likelihood of the application! *)
  if upper_bound <= 0. then () else 
    match argument_types with
    | [] -> (* not a function so we don't need any applications *)
      if lower_bound < 0. && 0. <= upper_bound then callBack f context 0.0 else ()
    | first_argument::later_arguments ->
      let first_argument = applyContext context first_argument in
      enumerate_programs g context first_argument environment 0. upper_bound
        (fun a k ll ->
           let a = Apply(f,a) in
           enumerate_applications g k later_arguments a environment (lower_bound+.ll) (upper_bound+.ll)
             (fun a k a_ll -> callBack a k (a_ll+.ll)))

