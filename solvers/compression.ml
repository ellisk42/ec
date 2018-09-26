open Core


open Gc

open Tower
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar

(* open Eg *)
open Versions

let verbose_compression = ref false;;

let inside_outside ~pseudoCounts g (frontiers : frontier list) =
  let summaries = frontiers |> List.map ~f:(fun f ->
      f.programs |> List.map ~f:(fun (p,l) ->
          let s = make_likelihood_summary g f.request p in
          (l, s))) in

  let update g =
    let weighted_summaries = summaries |> List.map ~f:(fun ss ->
        let log_weights = ss |> List.map ~f:(fun (l,s) ->
            l+. summary_likelihood g s) in
        let z = lse_list log_weights in
        List.map2_exn log_weights ss ~f:(fun lw (_,s) -> (exp (lw-.z),s))) |>
                             List.concat
    in

    let s = mix_summaries weighted_summaries in
    let possible p = Hashtbl.fold ~init:0. s.normalizer_frequency  ~f:(fun ~key ~data accumulator ->
        if List.mem ~equal:program_equal key p then accumulator+.data else accumulator)
    in
    let actual p = match Hashtbl.find s.use_frequency p with
      | None -> 0.
      | Some(f) -> f
    in

    {logVariable = log (actual (Index(0)) +. pseudoCounts) -. log (possible (Index(0)) +. pseudoCounts);
     library = g.library |> List.map ~f:(fun (p,t,_,u) ->
         let l = log (actual p +. pseudoCounts) -. log (possible p +. pseudoCounts) in
       (p,t,l,u))}
  in
  let g = update g in
  (g,
   summaries |> List.map ~f:(fun ss ->
     ss |> List.map ~f:(fun (l,s) -> l+. summary_likelihood g s) |> lse_list) |> fold1 (+.))
    
    

  

exception EtaExpandFailure;;

let eta_long request e =
  let context = ref empty_context in

  let make_long e request =
    if is_arrow request then Some(Abstraction(Apply(shift_free_variables 1 e, Index(0)))) else None
  in 

  let rec visit request environment e = match e with
    | Abstraction(b) when is_arrow request ->
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
          List.fold_left (List.zip_exn xs xt) ~init:f ~f:(fun return_value (x,t) ->
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
    | Index(i) -> Index(d + List.Assoc.find_exn ~equal:(=) mapping (i - d))
    | Abstraction(b) -> Abstraction(visit (d + 1) b)
    | Apply(f,x) -> Apply(visit d f,
                          visit d x)
    | Primitive(_,_,_) | Invented(_,_) as e -> e
  in
  
  let renamed = visit 0 i in
  let abstracted = List.fold_right mapping ~init:renamed ~f:(fun _ e -> Abstraction(e)) in
  make_invention abstracted

let rewrite_with_invention i =
  (* Raises EtaExpandFailure if this is not successful *)
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
    let e' = visit e |> eta_long request in
    assert (program_equal
              (beta_normal_form ~reduceInventions:true e)
              (beta_normal_form ~reduceInventions:true e'));
    e'

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

  
  
let compression_step ~structurePenalty ~aic ~pseudoCounts ?arity:(arity=3) ~bs ~topI ~topK g frontiers =

  let restrict frontier =
    let restriction =
      frontier.programs |> List.map ~f:(fun (p,ll) ->
          (ll+.likelihood_under_grammar g frontier.request p,p,ll)) |>
      sort_by (fun (posterior,_,_) -> 0.-.posterior) |>
      List.map ~f:(fun (_,p,ll) -> (p,ll))
    in
    {request=frontier.request; programs=List.take restriction topK}
  in

  let original_frontiers = frontiers in
  let frontiers = ref (List.map ~f:restrict frontiers) in
  
  let score g frontiers =
    let g,ll = inside_outside ~pseudoCounts g frontiers in

    let production_size = function
      | Primitive(_,_,_) -> 1
      | Invented(_,e) -> program_size e
      | _ -> raise (Failure "Element of grammar is neither primitive nor invented")
    in 

    (g,
     ll-. aic*.(List.length g.library |> Float.of_int) -.
     structurePenalty*.(g.library |> List.map ~f:(fun (p,_,_,_) ->
         production_size p) |> sum |> Float.of_int))
  in

  let v = new_version_table() in
  let cost_table = empty_cost_table v in

  (* calculate candidates *)
  let frontier_indices : int list list = time_it "calculated version spaces" (fun () ->
      !frontiers |> List.map ~f:(fun f -> f.programs |> List.map ~f:(fun (p,_) ->
          incorporate v p |> n_step_inversion v ~n:arity))) in
  

  let candidates : int list = time_it "proposed candidates" (fun () ->
      let reachable : int list list = frontier_indices |> List.map ~f:(reachable_versions v) in
      let inhabitants : int list list = reachable |> List.map ~f:(fun indices ->
          List.concat_map ~f:(snd % minimum_cost_inhabitants cost_table) indices |>
          List.dedup_and_sort ~compare:(-)) in
      inhabitants |> List.concat |> occurs_multiple_times)
  in
  let candidates = candidates |> List.filter ~f:(nontrivial % List.hd_exn % extract v) in
  Printf.eprintf "Got %d candidates.\n" (List.length candidates);

  match candidates with
  | [] -> None
  | _ -> 

    let ranked_candidates = time_it "beamed version spaces" (fun () ->
        beam_costs ~ct:cost_table ~bs candidates frontier_indices)
    in
    let ranked_candidates = List.take ranked_candidates topI in

    let try_invention_and_rewrite_frontiers (i : int) =
      let invention_source = extract v i |> singleton_head in
      try
        let new_primitive = invention_source |> normalize_invention in
        if List.mem ~equal:program_equal (grammar_primitives g) new_primitive then raise DuplicatePrimitive;
        let new_grammar =
          uniform_grammar (new_primitive :: (grammar_primitives g))
        in 

        let rewriter = rewrite_with_invention invention_source in
        (* Extract the frontiers in terms of the new primitive *)
        let new_cost_table = empty_cost_table v in
        let new_frontiers = List.map !frontiers
            ~f:(fun frontier ->
                let programs' =
                  List.map frontier.programs ~f:(fun (originalProgram, ll) ->
                      let index = incorporate v originalProgram |> n_step_inversion v ~n:arity in
                      let program = minimum_cost_inhabitants new_cost_table ~given:(Some(i)) index |> snd |> 
                                    List.hd_exn |> extract v |> singleton_head in
                      let program' =
                        try rewriter frontier.request program
                        with EtaExpandFailure -> originalProgram
                      in
                      (program',ll))
                in 
                {request=frontier.request;
                 programs=programs'})
        in
        let new_grammar,s = score new_grammar new_frontiers in
        (s,new_grammar,new_frontiers)
      with UnificationFailure | DuplicatePrimitive -> (* ill-typed / duplicatedprimitive *)
        (Float.neg_infinity, g, !frontiers)
    in

    let _,initial_score = score g !frontiers in
    Printf.eprintf "Initial score: %f\n" initial_score;


    let best_score,g',frontiers',best_index =
      time_it (Printf.sprintf "Evaluated top-%d candidates" topI) (fun () -> 
      ranked_candidates |> List.map ~f:(fun (c,i) ->
          let source = extract v i |> singleton_head in
          let source = normalize_invention source in

          let s,g',frontiers' = try_invention_and_rewrite_frontiers i in
          if !verbose_compression then
            (Printf.eprintf "Invention %s : %s\nDiscrete score %f\n\tContinuous score %f\n"
              (string_of_program source)
              (closed_inference source |> string_of_type)
              c s;
             frontiers' |> List.iter ~f:(fun f -> Printf.eprintf "%s\n" (string_of_frontier f));
             Printf.eprintf "\n"; flush_everything());
          (s,g',frontiers',i))
      |> minimum_by (fun (s,_,_,_) -> -.s)) in

    if best_score < initial_score then
      (Printf.eprintf "No improvement possible.\n"; None)
    else
      (let new_primitive = grammar_primitives g' |> List.hd_exn in
       Printf.eprintf "Improved score to %f (dScore=%f) w/ new primitive\n\t%s : %s\n"
         best_score (best_score-.initial_score)
         (string_of_program new_primitive) (closed_inference new_primitive |> canonical_type |> string_of_type);
       flush_everything();
       (* Rewrite the entire frontiers *)
       frontiers := original_frontiers;
       let _,g'',frontiers'' = time_it "rewrote all of the frontiers" (fun () ->
           try_invention_and_rewrite_frontiers best_index)
       in

       Some(g'',frontiers''))
;;

let compression_loop
    ~structurePenalty ~aic ~topK ~pseudoCounts ?arity:(arity=3) ~bs ~topI g frontiers =

  let rec loop g frontiers = 
    match compression_step ~structurePenalty ~topK ~aic ~pseudoCounts ~arity ~bs ~topI g frontiers with
    | None -> g, frontiers
    | Some(g',frontiers') -> loop g' frontiers'
  in
  loop g frontiers
  
  




  
  

let () =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let j = Yojson.Basic.from_channel Pervasives.stdin in
  let g = j |> member "DSL" |> deserialize_grammar |> strip_grammar in
  let topK = j |> member "topK" |> to_int in
  let topI = j |> member "topI" |> to_int in
  let bs = j |> member "bs" |> to_int in
  let arity = j |> member "arity" |> to_int in
  let aic = j |> member "aic" |> to_float in
  let pseudoCounts = j |> member "pseudoCounts" |> to_float in
  let structurePenalty = j |> member "structurePenalty" |> to_float in

  verbose_compression := (try
      j |> member "verbose" |> to_bool
    with _ -> false);

  let frontiers = j |> member "frontiers" |> to_list |> List.map ~f:deserialize_frontier in 

  
  

  (* let ps = [(\* "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (car $0) ($1 (cdr (cdr $0)))))))))"; *\) *)
  (*           (\*     "(lambda (fix1 $0 (lambda (lambda (if (eq? 0 $0) empty (cons (- 0 $0) ($1 (+ 1 $0))))))))"; *\) *)
  (*           "(lambda (fold $0 empty (lambda (lambda (cons (+ (+ 5 5) (+ $1 $1)) $0)))))"; *)
  (*           "(lambda (fold $0 empty (lambda (lambda (cons (- 0 $1) $0)))))"; *)
  (*           "(lambda (fold $0 empty (lambda (lambda (cons (+ $1 $1) $0)))))"; *)
            
  (*   "(lambda (- $0 (+ 4 4)))";"(lambda (+ $0 $0))"; *)
  (* ] |> List.map ~f:(compose get_some parse_program) |> List.map ~f:strip_primitives *)
  (* in *)

  
  (* let g = primitive_grammar (ps |> List.map ~f:program_subexpressions |> List.concat |> List.filter *)
  (*                              ~f:is_primitive |> List.dedup_and_sort ~compare:compare_program) in *)
  (* let frontiers = (ps |> List.map ~f:(fun p -> {request=closed_inference p;programs=[(p,0.)]})) in *)
  let g, frontiers = compression_loop ~topK ~aic ~structurePenalty ~pseudoCounts ~arity ~topI ~bs g frontiers in

  let j = `Assoc(["DSL",serialize_grammar g;
                  "frontiers",`List(frontiers |> List.map ~f:serialize_frontier)])
  in
  pretty_to_string j |> print_string
