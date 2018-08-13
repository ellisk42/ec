open Core


open Gc
    
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar

(* open Eg *)
open Versions

exception EtaExpandFailure;;

let eta_long request e =
  let context = ref empty_context in

  let make_long e request =
    if is_arrow request then Some(Abstraction(Apply(shift_free_variables 1 e, Index(0)))) else None
  in 

  let rec visit request environment e = match e with
    | Abstraction(b) when (is_arrow request) ->
      Abstraction(visit (right_of_arrow request) (left_of_arrow request :: environment) b)
    | Abstraction(_) -> raise EtaExpandFailure
    | _ -> match make_long e request with
      | Some(e') -> visit request environment e'
      | None -> (* match e with *)
        (* | Index(i) -> (unify' context request (List.nth_exn environment i); e) *)
        (* | Primitive(t,_,_) | Invented(t,_) -> *)
        (*   (let t = instantiate_type' context t in *)
        (*    unify' context t request; *)
        (*    e) *)
        (* | Abstraction(_) -> assert false *)
        (* | Apply(_,_) -> *)
        let f,xs = application_parse e in
        let ft = match f with
          | Index(i) -> environment $$ i |> applyContext' context
          | Primitive(t,_,_) | Invented(t,_) -> instantiate_type' context t
          | Abstraction(_) -> assert false (* not in beta long form *)
          | Apply(_,_) -> assert false
        in
        unify' context request (return_of_type ft);
        let ft = applyContext' context ft in
        let xt = arguments_of_type ft in
        if List.length xs <> List.length xt then raise EtaExpandFailure else
          List.fold_right (List.zip_exn xs xt) ~init:f ~f:(fun (x,t) return_value ->
              Apply(return_value,
                    visit (applyContext' context t) environment x))
  in

  let e' = visit request [] e in
  assert (tp_eq
            (e |> closed_inference |> canonical_type)
            (e' |> closed_inference |> canonical_type));
  e'
;;

let normalize_invention i =
  let mapping = free_variables i |> List.dedup_and_sort ~compare:(-) |> List.mapi ~f:(fun i v -> (v,i)) in

  let rec visit d = function
    | Index(i) when i < d -> Index(i)
    | Index(i) -> Index(List.Assoc.find_exn ~equal:(=) mapping (i - d))
    | Abstraction(b) -> Abstraction(visit (d + 1) b)
    | Apply(f,x) -> Apply(visit d f,
                          visit d x)
    | Primitive(_,_,_) | Invented(_,_) as e -> e
  in
  
  let renamed = visit 0 i in
  List.fold_right mapping ~init:renamed ~f:(fun _ e -> Abstraction(e)) |> make_invention

let rewrite_with_invention i =
  let mapping = free_variables i |> List.dedup_and_sort ~compare:(-) |> List.mapi ~f:(fun i v -> (i,v)) in
  let closed = normalize_invention i in
  (* FIXME : no idea whether I got this correct or not... *)
  let applied_invention = List.fold_left ~init:closed
      (List.range ~start:`exclusive ~stop:`inclusive ~stride:(-1) (List.length mapping) 0)
      ~f:(fun e i -> Apply(e,Index(List.Assoc.find_exn ~equal:(=) mapping i)))
  in

  let rec visit e =
    if program_equal e i then applied_invention else
      match e with
      | Apply(f,x) -> Apply(visit f, visit x)
      | Abstraction(b) -> Abstraction(visit b)
      | Index(_) | Primitive(_,_,_) | Invented(_,_) -> e
  in
  fun request e -> 
    try visit e |> eta_long request
    with EtaExpandFailure -> e

let nontrivial e =
  let indices = ref [] in
  let duplicated_indices = ref 0 in
  let primitives = ref 0 in
  let rec visit d = function
    | Index(i) ->
      let i = i - d in
      if List.mem ~equal:(=) !indices i
      then incr duplicated_indices
      else indices := i :: !indices
    | Apply(f,x) -> (visit d f; visit d x)
    | Abstraction(b) -> visit (d + 1) b
    | Primitive(_,_,_) | Invented(_,_) -> incr primitives
  in
  visit 0 e;
  !primitives > 1 || !primitives = 1 && !duplicated_indices > 0
;;

  
  
let compression_step ?arity:(arity=3) ~bs g frontiers =
  let v = new_version_table() in
  let frontier_indices = time_it "calculated version spaces" (fun () ->
      frontiers |> List.map ~f:(fun f -> f.programs |> List.map ~f:(fun (p,_) ->
          incorporate v p |> n_step_inversion v ~n:arity))) in
  
  let cost_table = empty_cost_table v in
  let candidates : int list = time_it "proposed candidates" (fun () ->
      let reachable : int list list = frontier_indices |> List.map ~f:(reachable_versions v) in
      let inhabitants : int list list = reachable |> List.map ~f:(fun indices ->
          List.concat_map ~f:(snd % minimum_cost_inhabitants cost_table) indices |>
          List.dedup_and_sort ~compare:(-)) in
      inhabitants |> List.concat |> occurs_multiple_times)
  in
  let candidates = candidates |> List.filter ~f:(nontrivial % List.hd_exn % extract v) in
  Printf.eprintf "Got %d candidates.\n" (List.length candidates);

  let ranked_candidates = time_it "beamed version spaces" (fun () ->
      beam_costs ~ct:cost_table ~bs candidates frontier_indices)
  in

  ranked_candidates |> List.iter ~f:(fun (c,i) ->
      let [i] = extract v i in
      Printf.eprintf "%f\t%s\n"
        c
        (string_of_program i))
;;

        
  
  




  
  

let _ =
  let ps = ["(lambda (fold $0 empty (lambda (lambda (cons (+ (+ 5 5) (+ $1 $1)) $0)))))";
            "(lambda (fold $0 empty (lambda (lambda (cons (- 0 $1) $0)))))";
            "(lambda (fold $0 empty (lambda (lambda (cons (+ $1 $1) $0)))))";
            "(lambda (+ $0 $0))";
            "(lambda (+ 4 4))";] |> List.map ~f:(compose get_some parse_program)
    in
  compression_step ~bs:25 () (ps |> List.map ~f:(fun p -> {request=magical();programs=[(p,0.)]}))
  (* let p' = parse_program "(+ 9 9)" |> get_some in *)
  (* let j = time_it "calculated versions base" (fun () -> p |> incorporate t |> recursive_inversion t |> recursive_inversion t  |> recursive_inversion t) in *)
  (* extract t j |> List.map ~f:(fun r -> *)
  (*     (\* Printf.printf "%s\n\t%s\n" (string_of_program r) *\) *)
  (*     (\*   (beta_normal_form r |> string_of_program); *\) *)
  (*     (\* flush_everything(); *\) *)
  (*     assert ((string_of_program p) = (beta_normal_form r |> string_of_program))); *)
  (* Printf.printf "Enumerated %d version spaces.\n" *)
  (*   (t.i2s.ra_occupancy) *)

