open Core

open [@warning "-33"] Physics
open [@warning "-33"] Pregex
open [@warning "-33"] Tower
open Utils
open Type
open Program
open Enumeration
open Grammar

(* open Eg *)
open Versions

let verbose_compression = ref false;;

(* If this is true, then we collect and report data on the sizes of the version spaces, for each program, and also for each round of inverse beta *)
let collect_data = ref false;;

let restrict ~topK g frontier =
  let restriction =
    frontier.programs |> List.map ~f:(fun (p,ll) ->
        (ll+.likelihood_under_grammar g frontier.request p,p,ll)) |>
    sort_by (fun (posterior,_,_) -> 0. -. posterior) |>
    List.map ~f:(fun (_,p,ll) -> (p,ll))
  in
  {request=frontier.request; programs=List.take restriction topK}


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

    {g with
     logVariable = log (actual (Index(0)) +. pseudoCounts) -. log (possible (Index(0)) +. pseudoCounts);
     library = g.library |> List.map ~f:(fun (p,t,_,u) ->
         let l = log (actual p +. pseudoCounts) -. log (possible p +. pseudoCounts) in
       (p,t,l,u))}
  in
  let g = update g in
  (g,
   summaries |> List.map ~f:(fun ss ->
     ss |> List.map ~f:(fun (l,s) -> l+. summary_likelihood g s) |> lse_list) |> fold1 (+.))


let grammar_induction_score ~aic ~structurePenalty ~pseudoCounts frontiers g =
  let g,ll = inside_outside ~pseudoCounts g frontiers in

  let production_size = function
    | Primitive(_,_,_) -> 1
    | Invented(_,e) -> begin
        (* Ignore illusory fix1/abstraction, it does not contribute to searching cost *)
        let e = recursively_get_abstraction_body e in
        match e with
        | Apply(Apply(Primitive(_,"fix1",_),i),b) -> (assert (is_index i); program_size b)
        | _ -> program_size e
      end
    | _ -> raise (Failure "Element of grammar is neither primitive nor invented")
  in

  (g,
   ll-. aic*.(List.length g.library |> Float.of_int) -.
   structurePenalty*.(g.library |> List.map ~f:(fun (p,_,_,_) ->
       production_size p) |> sum |> Float.of_int))



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
          let xs' =
            List.map2_exn xs xt ~f:(fun x t -> visit (applyContext' context t) environment x)
          in
          List.fold_left xs' ~init:f ~f:(fun return_value x ->
              Apply(return_value,x))
  in

  let e' = visit request [] e in

  assert (tp_eq
            (e |> closed_inference |> canonical_type)
            (e' |> closed_inference |> canonical_type));
  e'
;;

let normalize_invention i =
  (* Raises UnificationFailure if i is not well typed *)
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
    try
      let e' = visit e |> eta_long request in
      assert (program_equal
                (beta_normal_form ~reduceInventions:true e)
                (beta_normal_form ~reduceInventions:true e'));
      e'
    with UnificationFailure -> begin
        if !verbose_compression then begin
          Printf.eprintf "WARNING: rewriting with invention gave ill typed term.\n";
          Printf.eprintf "Original:\t\t%s\n" (e |> string_of_program);
          Printf.eprintf "Original:\t\t%s\n" (e |> beta_normal_form ~reduceInventions:true |> string_of_program);
          Printf.eprintf "Rewritten:\t\t%s\n" (visit e |> string_of_program);
          Printf.eprintf "Rewritten:\t\t%s\n" (visit e |> beta_normal_form ~reduceInventions:true |> string_of_program);
          Printf.eprintf "Going to proceed as if the rewrite had failed - but look into this because it could be a bug.\n";
          flush_everything()
        end;
        let normal_original = e |> beta_normal_form ~reduceInventions:true in
        let normal_rewritten = e |> visit |> beta_normal_form ~reduceInventions:true in
        assert (program_equal normal_original normal_rewritten);
        raise EtaExpandFailure
    end


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


type worker_command =
  | Rewrite of program list
  | RewriteEntireFrontiers of program
  | KillWorker
  | FinalFrontier of program
  | BatchedRewrite of program list

let compression_worker connection ~inline ~arity ~bs ~topK g frontiers =
  let context = Zmq.Context.create() in
  let socket = Zmq.Socket.create context Zmq.Socket.req in
  Zmq.Socket.connect socket connection;
  let send data = Zmq.Socket.send socket (Marshal.to_string data []) in
  let receive() = Marshal.from_string (Zmq.Socket.recv socket) 0 in


  let original_frontiers = frontiers in
  let frontiers = ref (List.map ~f:(restrict ~topK g) frontiers) in

  let v = new_version_table() in

  (* calculate candidates from the frontiers we can see *)
  let frontier_indices : int list list = time_it ~verbose:!verbose_compression
      "(worker) calculated version spaces" (fun () ->
      !frontiers |> List.map ~f:(fun f -> f.programs |> List.map ~f:(fun (p,_) ->
              incorporate v p |> n_step_inversion v ~inline ~n:arity))) in
  if !collect_data then begin
    List.iter2_exn !frontiers frontier_indices ~f:(fun frontier indices ->
        List.iter2_exn (frontier.programs) indices ~f:(fun (p,_) index ->
            let rec program_size = function
              | Apply(f,x) -> 1 + program_size f + program_size x
              | Abstraction(b) -> 1 + program_size b
              | Index(_) | Invented(_,_) | Primitive(_,_,_) -> 1
            in
            let rec program_height = function
              | Apply(f,x) -> 1 + (max (program_height f) (program_height x))
              | Abstraction(b) -> 1 + program_height b
              | Index(_) | Invented(_,_) | Primitive(_,_,_) -> 1
            in
            Printf.eprintf "DATA\t%s\tsize=%d\theight=%d\t|vs|=%d\t|[vs]|=%f\n" (string_of_program p)
              (program_size p) (program_height p)
              (reachable_versions v [index] |> List.length)
              (log_version_size v index)
          ))
  end;
  if !verbose_compression then
    Printf.eprintf "(worker) %d distinct version spaces enumerated; %d accessible vs size; vs log sizes: %s\n"
      v.i2s.ra_occupancy
      (frontier_indices |> List.concat |> reachable_versions v |> List.length)
      (frontier_indices |> List.concat |> List.map ~f:(Float.to_string % log_version_size v)
       |> join ~separator:"; ");

  let v, frontier_indices = garbage_collect_versions ~verbose:!verbose_compression v frontier_indices in
  Gc.compact();

  let cost_table = empty_cost_table v in

  (* pack the candidates into a version space for efficiency *)
  let candidate_table = new_version_table() in
  let candidates : int list list = time_it ~verbose:!verbose_compression "(worker) proposed candidates"
      (fun () ->
      let reachable : int list list = frontier_indices |> List.map ~f:(reachable_versions v) in
      let inhabitants : int list list = reachable |> List.map ~f:(fun indices ->
          List.concat_map ~f:(snd % minimum_cost_inhabitants cost_table) indices |>
          List.dedup_and_sort ~compare:(-) |>
          List.map ~f:(List.hd_exn % extract v) |>
          List.filter ~f:nontrivial |>
          List.map ~f:(incorporate candidate_table)) in
          inhabitants)
  in
  if !verbose_compression then Printf.eprintf "(worker) Total candidates: [%s] = %d, packs into %d vs\n"
      (candidates |> List.map ~f:(Printf.sprintf "%d" % List.length) |> join ~separator:";")
      (candidates |> List.map ~f:List.length |> sum)
      (candidate_table.i2s.ra_occupancy);
  flush_everything();

  (* relay this information to the master, whose job it is to pool the candidates *)
  send (candidates,candidate_table.i2s);
  let [@warning "-26"] candidate_table = () in
  let candidates : program list = receive() in
  let candidates : int list = candidates |> List.map ~f:(incorporate v) in

  if !verbose_compression then
    (Printf.eprintf "(worker) Got %d candidates.\n" (List.length candidates);
     flush_everything());

  let candidate_scores : float list = time_it ~verbose:!verbose_compression "(worker) beamed version spaces"
      (fun () ->
      beam_costs' ~ct:cost_table ~bs candidates frontier_indices)
  in

  send candidate_scores;
   (* I hope that this leads to garbage collection *)
  let [@warning "-26"] candidate_scores = ()
  and [@warning "-26"] cost_table = ()
  in
  Gc.compact();

  let rewrite_frontiers invention_source =
    time_it ~verbose:!verbose_compression "(worker) rewrote frontiers" (fun () ->
        time_it ~verbose:!verbose_compression "(worker) gc during rewrite" Gc.compact;
        let intersectionTable = Some(Hashtbl.Poly.create()) in
        let i = incorporate v invention_source in
        let rewriter = rewrite_with_invention invention_source in
        (* Extract the frontiers in terms of the new primitive *)
        let new_cost_table = empty_cheap_cost_table v in
        let new_frontiers = List.map !frontiers
            ~f:(fun frontier ->
                let programs' =
                  List.map frontier.programs ~f:(fun (originalProgram, ll) ->
                      let index = incorporate v originalProgram |> n_step_inversion ~inline v ~n:arity in
                      let program =
                        minimal_inhabitant ~intersectionTable new_cost_table ~given:(Some(i)) index |> get_some
                      in
                      let program' =
                        try rewriter frontier.request program
                        with EtaExpandFailure -> originalProgram
                      in
                      (program',ll))
                in
                {request=frontier.request;
                 programs=programs'})
        in
        new_frontiers)
  in

  let batched_rewrite inventions =
    time_it ~verbose:!verbose_compression "(worker) batched rewrote frontiers" (fun () ->
        Gc.compact();
        let invention_indices : int list = inventions |> List.map ~f:(incorporate v) in
        let frontier_indices : int list list =
          !frontiers |> List.map ~f:(fun f ->
              f.programs |> List.map ~f:(n_step_inversion ~inline v ~n:arity % incorporate v % fst))
        in
        clear_dynamic_programming_tables v;
        let refactored = batched_refactor ~ct:(empty_cost_table v) invention_indices frontier_indices in
        Gc.compact();
        List.map2_exn refactored inventions ~f:(fun new_programs invention ->
            let rewriter = rewrite_with_invention invention in
            List.map2_exn new_programs !frontiers ~f:(fun new_programs frontier ->
                let programs' =
                  List.map2_exn new_programs frontier.programs ~f:(fun program (originalProgram, ll) ->
                      if not (program_equal
                                (beta_normal_form ~reduceInventions:true program)
                                (beta_normal_form ~reduceInventions:true originalProgram)) then
                        (Printf.eprintf "FATAL: %s refactored into %s\n"
                           (string_of_program originalProgram)
                           (string_of_program program);
                         Printf.eprintf "This has never occurred before! Definitely send this to Kevin, if this occurs it is a terrifying bug.\n";
                        assert (false));
                      let program' =
                        try rewriter frontier.request program
                        with EtaExpandFailure -> originalProgram
                      in
                      (program',ll))
                in
                {request=frontier.request;
                 programs=programs'})))
  in

  let final_rewrite invention =
    (* As our last act, free as much memory as we can *)
    deallocate_versions v; Gc.compact();

    (* exchanging time for memory - invert everything again *)
    frontiers := original_frontiers;
    let v = new_version_table() in
    let frontier_inversions = Hashtbl.Poly.create() in
    time_it ~verbose:!verbose_compression "(worker) did final inversion" (fun () ->
        !frontiers |> List.iter ~f:(fun f ->
            f.programs |> List.iter ~f:(fun (p,_) ->
                Hashtbl.set frontier_inversions
                  ~key:(incorporate v p)
                  ~data:(n_step_inversion ~inline v ~n:arity (incorporate v p)))));
    clear_dynamic_programming_tables v; Gc.compact();

    let i = incorporate v invention in
    let new_cost_table = empty_cheap_cost_table v in
    time_it ~verbose:!verbose_compression "(worker) did final refactor" (fun () ->
        List.map !frontiers
          ~f:(fun frontier ->
              let programs' =
                List.map frontier.programs ~f:(fun (originalProgram, ll) ->
                    let index = Hashtbl.find_exn frontier_inversions (incorporate v originalProgram) in
                    let program =
                      minimal_inhabitant new_cost_table ~given:(Some(i)) index |> get_some
                    in
                    let program' =
                      try rewrite_with_invention invention frontier.request program
                      with EtaExpandFailure -> originalProgram
                    in
                    (program',ll))
              in
              {request=frontier.request;
               programs=programs'}))
  in

  while true do
    match receive() with
    | Rewrite(i) -> send (i |> List.map ~f:rewrite_frontiers)
    | RewriteEntireFrontiers(i) ->
      (frontiers := original_frontiers;
       send (rewrite_frontiers i))
    | BatchedRewrite(inventions) -> send (batched_rewrite inventions)
    | FinalFrontier(invention) ->
      (frontiers := original_frontiers;
       send (final_rewrite invention);
       Gc.compact())
    | KillWorker ->
       (Zmq.Socket.close socket;
        Zmq.Context.terminate context;
       exit 0)
  done;;

let compression_step_master ~inline ~nc ~structurePenalty ~aic ~pseudoCounts ?arity:(arity=3) ~bs ~topI ~topK g frontiers =

  let sockets = ref [] in
  let timestamp = Time.now() |> Time.to_filename_string ~zone:Time.Zone.utc in
  let fork_worker frontiers =
    let p = List.length !sockets in
    let address = Printf.sprintf "ipc:///tmp/compression_ipc_%s_%d" timestamp p in
    sockets := !sockets @ [address];

    match Unix.fork() with
    | `In_the_child -> compression_worker address ~arity ~bs ~topK ~inline g frontiers
    | _ -> ()
  in

  if !verbose_compression then ignore(Unix.system "ps aux|grep compression 1>&2" : Core.Unix.Exit_or_signal.t);

  let divide_work_fairly nc xs =
    let nt = List.length xs in
    let base_count = nt/nc in
    let residual = nt - base_count*nc in
    let rec partition residual xs =
      let this_count =
        base_count + (if residual > 0 then 1 else 0)
      in
      match xs with
      | [] -> []
      | _ :: _ ->
        let prefix, suffix = List.split_n xs this_count in
        prefix :: partition (residual - 1) suffix
    in
    partition residual xs
  in
  let start_time = Time.now () in
  divide_work_fairly nc frontiers |> List.iter ~f:fork_worker;

  (* Now that we have created the workers, we can make our own sockets *)
  let context = Zmq.Context.create() in
  let sockets = !sockets |> List.map ~f:(fun address ->
      let socket = Zmq.Socket.create context Zmq.Socket.rep in
      Zmq.Socket.bind socket address;
      socket)
  in
  let send data =
    let data = Marshal.to_string data [] in
    sockets |> List.iter ~f:(fun socket -> Zmq.Socket.send socket data)
  in
  let receive socket = Marshal.from_string (Zmq.Socket.recv socket) 0 in
  let finish() =
    send KillWorker;
    sockets |> List.iter ~f:(fun s -> Zmq.Socket.close s);
    Zmq.Context.terminate context
  in



  let candidates : program list list = sockets |> List.map ~f:(fun s ->
      let candidate_message : (int list list)*(vs ra) = receive s in
      let (candidates, candidate_table) = candidate_message in
      let candidate_table = {(new_version_table()) with i2s=candidate_table} in
      candidates |> List.map ~f:(List.map ~f:(singleton_head % extract candidate_table))) |> List.concat in
  let candidates : program list = occurs_multiple_times (List.concat candidates) in
  Printf.eprintf "Total number of candidates: %d\n" (List.length candidates);
  Printf.eprintf "Constructed version spaces and coalesced candidates in %s.\n"
    (Time.diff (Time.now ()) start_time |> Time.Span.to_string);
  flush_everything();

  send candidates;

  let candidate_scores : float list list =
    sockets |> List.map ~f:(fun s -> let ss : float list = receive s in ss)
  in
  if !verbose_compression then (Printf.eprintf "(master) Received worker beams\n"; flush_everything());
  let candidates : program list =
    candidate_scores |> List.transpose_exn |>
    List.map ~f:(fold1 (+.)) |> List.zip_exn candidates |>
    List.sort ~compare:(fun (_,s1) (_,s2) -> Float.compare s1 s2) |> List.map ~f:fst
  in
  let candidates = List.take candidates topI in
  let candidates = candidates |> List.filter ~f:(fun candidate ->
      try
        let candidate = normalize_invention candidate in
        not (List.mem ~equal:program_equal (grammar_primitives g) candidate)
      with UnificationFailure -> false) (* not well typed *)
  in
  Printf.eprintf "Trimmed down the beam, have only %d best candidates\n"
    (List.length candidates);
  flush_everything();

  match candidates with
  | [] -> (finish(); None)
  | _ ->

  (* now we have our final list of candidates! *)
  (* ask each of the workers to rewrite w/ each candidate *)
  send @@ BatchedRewrite(candidates);
  (* For each invention, the full rewritten frontiers *)
  let new_frontiers : frontier list list =
    time_it "Rewrote topK" (fun () ->
        sockets |> List.map ~f:receive |> List.transpose_exn |> List.map ~f:List.concat)
  in
  assert (List.length new_frontiers = List.length candidates);

  let score frontiers candidate =
    let new_grammar = uniform_grammar (normalize_invention candidate :: grammar_primitives g) in
    let g',s = grammar_induction_score ~aic ~pseudoCounts ~structurePenalty frontiers new_grammar in
    if !verbose_compression then
      (let source = normalize_invention candidate in
       Printf.eprintf "Invention %s : %s\n\tContinuous score %f\n"
         (string_of_program source)
         (closed_inference source |> string_of_type)
         s;
       frontiers |> List.iter ~f:(fun f -> Printf.eprintf "%s\n" (string_of_frontier f));
       Printf.eprintf "\n"; flush_everything());
    (g',s)
  in

  let _,initial_score = grammar_induction_score ~aic ~structurePenalty ~pseudoCounts
      (frontiers |> List.map ~f:(restrict ~topK g)) g
  in
  Printf.eprintf "Initial score: %f\n" initial_score;

  let (g',best_score), best_candidate = time_it "Scored candidates" (fun () ->
      List.map2_exn candidates new_frontiers ~f:(fun candidate frontiers ->
          (score frontiers candidate, candidate)) |> minimum_by (fun ((_,s),_) -> -.s))
  in
  if Float.(<) best_score initial_score then
      (Printf.eprintf "No improvement possible.\n"; finish(); None)
    else
      (let new_primitive = grammar_primitives g' |> List.hd_exn in
       Printf.eprintf "Improved score to %f (dScore=%f) w/ new primitive\n\t%s : %s\n"
         best_score (best_score-.initial_score)
         (string_of_program new_primitive) (closed_inference new_primitive |> canonical_type |> string_of_type);
       flush_everything();
       (* Rewrite the entire frontiers *)
       let frontiers'' : frontier list = time_it "rewrote all of the frontiers" (fun () ->
           send @@ FinalFrontier(best_candidate);
           sockets |> List.map ~f:receive |> List.concat)
       in
       finish();
       let g'' = inside_outside ~pseudoCounts g' frontiers'' |> fst in
       Some(g'',frontiers''))









let compression_step ~inline ~structurePenalty ~aic ~pseudoCounts ?arity:(arity=3) ~bs ~topI ~topK g frontiers =

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
    grammar_induction_score ~aic ~pseudoCounts ~structurePenalty frontiers g
  in

  let v = new_version_table() in
  let cost_table = empty_cost_table v in

  (* calculate candidates *)
  let frontier_indices : int list list = time_it "calculated version spaces" (fun () ->
      !frontiers |> List.map ~f:(fun f -> f.programs |> List.map ~f:(fun (p,_) ->
          incorporate v p |> n_step_inversion ~inline v ~n:arity))) in


  let candidates : int list = time_it "proposed candidates" (fun () ->
      let reachable : int list list = frontier_indices |> List.map ~f:(reachable_versions v) in
      let inhabitants : int list list = reachable |> List.map ~f:(fun indices ->
          List.concat_map ~f:(snd % minimum_cost_inhabitants cost_table) indices |>
          List.dedup_and_sort ~compare:(-)) in
      inhabitants |> List.concat |> occurs_multiple_times)
  in
  let candidates = candidates |> List.filter ~f:(fun candidate ->
      let candidate = List.hd_exn (extract v candidate) in
      try (ignore(normalize_invention candidate : program); nontrivial candidate)
      with UnificationFailure -> false)
  in
  Printf.eprintf "Got %d candidates.\n" (List.length candidates);

  match candidates with
  | [] -> None
  | _ ->

    let ranked_candidates = time_it "beamed version spaces" (fun () ->
        beam_costs ~ct:cost_table ~bs candidates frontier_indices)
    in
    let ranked_candidates = List.take ranked_candidates topI in

    let try_invention_and_rewrite_frontiers (i : int) =
      Gc.compact();
      let invention_source = extract v i |> singleton_head in
      try
        let new_primitive = invention_source |> normalize_invention in
        if List.mem ~equal:program_equal (grammar_primitives g) new_primitive then raise DuplicatePrimitive;
        let new_grammar =
          uniform_grammar (new_primitive :: (grammar_primitives g))
        in

        let rewriter = rewrite_with_invention invention_source in
        (* Extract the frontiers in terms of the new primitive *)
        let new_cost_table = empty_cheap_cost_table v in
        let new_frontiers = List.map !frontiers
            ~f:(fun frontier ->
                let programs' =
                  List.map frontier.programs ~f:(fun (originalProgram, ll) ->
                      let index = incorporate v originalProgram |> n_step_inversion v ~inline ~n:arity in
                      let program = minimal_inhabitant new_cost_table ~given:(Some(i)) index |> get_some in
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

    let best_score,g',_frontiers',best_index =
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
             frontiers' |> List.iter ~f:(fun f ->
                 let f = string_of_frontier f in
                 if String.is_substring ~substring:(string_of_program source) f then
                   Printf.eprintf "%s\n" f);
             Printf.eprintf "\n"; flush_everything());
          (s,g',frontiers',i))
      |> minimum_by (fun (s,_,_,_) -> -.s)) in

    if Float.(<) best_score initial_score then
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

let export_compression_checkpoint ~nc ~structurePenalty ~aic ~topK ~pseudoCounts ?arity:(arity=3) ~bs ~topI g frontiers =
  let timestamp = Time.now() |> Time.to_filename_string ~zone:Time.Zone.utc in
  let fn = Printf.sprintf "compressionMessages/%s" timestamp in

  let open Yojson.Basic in

  let j : Yojson.Basic.t =
    `Assoc(["DSL", serialize_grammar g;
            "topK", `Int(topK);
            "topI", `Int(topI);
            "bs", `Int(bs);
            "arity", `Int(arity);
            "pseudoCounts", `Float(pseudoCounts);
            "structurePenalty", `Float(structurePenalty);
            "verbose", `Bool(true);
            "CPUs", `Int(nc);
            "aic", `Float(aic);
            "frontiers", `List(frontiers |> List.map ~f:serialize_frontier)])
  in
  to_file fn j;
  Printf.eprintf "Exported compression checkpoint to %s\n" fn
;;


let compression_loop
    ?nc:(nc=1) ~structurePenalty ~inline ~aic ~topK ~pseudoCounts ?arity:(arity=3) ~bs ~topI ~iterations g frontiers =

  let find_new_primitive old_grammar new_grammar =
    new_grammar |> grammar_primitives |> List.filter ~f:(fun p ->
        not (List.mem ~equal:program_equal (old_grammar |> grammar_primitives) p)) |>
    singleton_head
  in
  let illustrate_new_primitive new_grammar primitive frontiers =
    let illustrations =
      frontiers |> List.filter_map ~f:(fun frontier ->
          let best_program = (restrict ~topK:1 new_grammar frontier).programs |> List.hd_exn |> fst in
          if List.mem ~equal:program_equal (program_subexpressions best_program) primitive then
            Some(best_program)
          else None)
    in
    Printf.eprintf "New primitive is used %d times in the best programs in each of the frontiers.\n"
      (List.length illustrations);
    Printf.eprintf "Here is where it is used:\n";
    illustrations |> List.iter ~f:(fun program -> Printf.eprintf "  %s\n" (string_of_program program))
  in

  let step = if nc = 1 then compression_step else compression_step_master ~nc in

  let rec loop ~iterations g frontiers =
    if iterations < 1 then
      (Printf.eprintf "Exiting ocaml compression because of iteration bound.\n";g, frontiers)
    else
      match time_it "Completed one step of memory consolidation"
              (fun () -> step ~inline ~structurePenalty ~topK ~aic ~pseudoCounts ~arity ~bs ~topI g frontiers)
      with
      | None -> g, frontiers
      | Some(g',frontiers') ->
        illustrate_new_primitive g' (find_new_primitive g g') frontiers';
        if !verbose_compression && iterations > 1 then
          export_compression_checkpoint ~nc ~structurePenalty ~aic ~topK ~pseudoCounts ~arity ~bs ~topI g' frontiers';
        flush_everything();
        loop ~iterations:(iterations - 1) g' frontiers'
  in
  time_it "completed ocaml compression" (fun () ->
      loop ~iterations g frontiers)
;;


let () =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let j =
    if Array.length (Sys.get_argv ()) > 1 then
      (assert (Array.length (Sys.get_argv ()) = 2);
       Yojson.Basic.from_file (Sys.get_argv ()).(1))
    else
      Yojson.Basic.from_channel Stdlib.stdin
  in
  let g = j |> member "DSL" |> deserialize_grammar |> strip_grammar in
  let topK = j |> member "topK" |> to_int in
  let topI = j |> member "topI" |> to_int in
  let bs = j |> member "bs" |> to_int in
  let arity = j |> member "arity" |> to_int in
  let aic = j |> member "aic" |> to_number in
  let pseudoCounts = j |> member "pseudoCounts" |> to_number in
  let structurePenalty = j |> member "structurePenalty" |> to_number in

  verbose_compression := (try
      j |> member "verbose" |> to_bool
                          with _ -> false);

  factored_substitution := (try
                              j |> member "factored_apply" |> to_bool
                            with _ -> false);
  if !factored_substitution then Printf.eprintf "Using experimental new factored representation of application version space.\n";

  collect_data := (try
                     j |> member "collect_data" |> to_bool
                   with _ -> false) ;
  if !collect_data then verbose_compression := true;


  let inline = (try
                  j |> member "inline" |> to_bool
                with _ -> true)
  in

  let nc =
    try j |> member "CPUs" |> to_int
    with _ -> 1
  in

  let iterations = try
      j |> member "iterations" |> to_int
    with _ -> 1000
  in
  Printf.eprintf "Compression backend will run for most %d iterations\n"
    iterations;
  flush_everything();

  let frontiers = j |> member "frontiers" |> to_list |> List.map ~f:deserialize_frontier in

  let g, frontiers =
    if Float.(>) aic 500. then
      (Printf.eprintf "AIC is very large (over 500), assuming you don't actually want to do DSL learning!";
       g, frontiers)
    else compression_loop ~inline ~iterations ~nc ~topK ~aic ~structurePenalty ~pseudoCounts ~arity ~topI ~bs g frontiers in


  let j = `Assoc(["DSL",serialize_grammar g;
                  "frontiers",`List(frontiers |> List.map ~f:serialize_frontier)])
  in
  pretty_to_string j |> print_string
